from __future__ import annotations
from typing import List, Tuple, cast, Optional, Any, Dict, Final, DefaultDict
import math, functools
from dataclasses import replace
from collections import defaultdict
from tinygrad.codegen.kernel import LocalBuffer, Kernel
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.dtype import dtypes, PtrDType, ImageDType
from tinygrad.ops import BufferOps, LazyOp, TernaryOps, ReduceOps, BinaryOps, UnaryOps, MemBuffer, get_lazyop_info
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.renderer import Program
from tinygrad.helpers import to_function_name, colored, DEBUG, getenv, prod

# TODO: this needs to be replaced, there shouldn't be variables in the shapetracker
from tinygrad.shape.symbolic import Variable, NumNode, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode
render_ops: Any = { NumNode: lambda self, ops, ctx: UOp.const(dtypes.int, self.b),
                    MulNode: lambda self, ops, ctx: self.a.render(ops, ctx)*self.b,
                    DivNode: lambda self, ops, ctx: self.a.render(ops, ctx)//self.b,
                    ModNode: lambda self, ops, ctx: self.a.render(ops, ctx)%self.b,
                    LtNode: lambda self, ops, ctx: self.a.render(ops, ctx).lt(self.b),
  Variable: lambda self,ops,ctx: ctx[self] if self in ctx else UOp(UOps.DEFINE_VAR, dtypes.int32, (), self),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a+b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a*b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

def variable_to_uop(x) -> UOp:
  if isinstance(x, int): return UOp.const(dtypes.int32, x)
  return x.render(render_ops)

"""
def _uop_view(view:View, idxs:List[UOp], vexpr:UOp) -> Tuple[UOp, UOp]:
  # TODO: dtypes.realint
  iexpr = variable_to_uop(view.offset)
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr = iexpr + idx*variable_to_uop(st)
    if m is not None:
      if m[0] != 0: vexpr = vexpr * idx.ge(variable_to_uop(m[0]))
      if m[1] != sh: vexpr = vexpr * idx.lt(variable_to_uop(m[1]))
  return iexpr, vexpr

def st_to_uops(st:ShapeTracker, idxs:List[UOp]) -> Tuple[UOp, UOp]:
  idx, valid = _uop_view(st.views[-1], idxs, UOp.const(dtypes.bool, True))
  for view in reversed(st.views[0:-1]):
    view = view.minify()
    acc, idxs = 1, []
    for d in reversed(view.shape):
      idxs.append((idx//acc)%variable_to_uop(d))
      acc *= variable_to_uop(d)
    idx, valid = _uop_view(view, idxs[::-1], valid)
  return idx, valid
"""

def st_to_uops(st:ShapeTracker, idxs:List[UOp]) -> Tuple[UOp, UOp]:
  fake_idxs = [Variable(f"__idx{i}", 0, s-1) for i,s in enumerate(st.shape)]
  idx, valid = st.expr_idxs(fake_idxs)
  ctx = dict(zip(fake_idxs, idxs))
  return idx.render(render_ops, ctx), valid.render(render_ops, ctx)

def get_grouped_dims(prefix, start_dim, local_dims, maxdim:int=0):
  local_idxs = loop_local_idxs = [UOp(UOps.SPECIAL, dtypes.int32, (), (i, f"{prefix}{start_dim+i}", s)) for i,s in enumerate((prod(local_dims[:-(maxdim-1)]),) + local_dims[-(maxdim-1):] if len(local_dims) > maxdim else local_dims)]  # noqa: E501
  if maxdim != 0 and len(local_dims) > maxdim:
    dd = local_idxs[0]
    nli = []
    for s in local_dims[:-(maxdim-1)]:
      nli.append(dd % s)
      dd //= s
    local_idxs = nli + local_idxs[-(maxdim-1):]
  return local_idxs, loop_local_idxs

def get_reduce_acc(reduceop:LazyOp):
  if reduceop.op is ReduceOps.SUM: return 0.0 if dtypes.is_float(reduceop.dtype) else 0
  if reduceop.op is ReduceOps.MAX:
    if dtypes.is_int(reduceop.dtype): return 0 if dtypes.is_unsigned(reduceop.dtype) else -2**(reduceop.dtype.itemsize*8-1)
    return -math.inf if dtypes.is_float(reduceop.dtype) else False

class Lowerer(Kernel):
  def to_uop(self, x:LazyOp) -> UOp:
    if uop:=self.uop_cache.get(x, None): return uop
    ret = self._to_uop(x)
    self.uop_cache[x] = ret
    return ret

  def _to_uop(self, x:LazyOp) -> UOp:
    if x.op in BufferOps:
      idx, valid = st_to_uops(x.arg.st, self.ridxs if x.op is BufferOps.LOAD and x.arg.idx == -1 else self.idxs)
      if x.op is BufferOps.CONST:
        return UOp.alu(TernaryOps.WHERE, valid, UOp.const(x.arg.dtype, x.arg.val), UOp.const(x.arg.dtype, 0))
      if x.arg.idx == -1:
        # TODO: this should come from somewhere else
        buf = self.local_buffer_uop
      else:
        buf = UOp(UOps.DEFINE_GLOBAL, x.arg.dtype if isinstance(x.arg.dtype, ImageDType) else PtrDType(x.arg.dtype), (),
                  (x.arg.idx, any(x.arg.idx == y.idx for y in self.outbufs)))
      if x.op is BufferOps.LOAD:
        barrier = (UOp(UOps.BARRIER, None, (self.to_uop(x.src[0]),)),) if len(x.src) else ()
        return UOp(UOps.LOAD, x.arg.dtype.scalar(), (buf, idx) + (valid, UOp.const(x.arg.dtype, 0)) + barrier)
      return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + (valid,))

    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op is UnaryOps.CAST: return UOp(UOps.CAST, x.arg.scalar(), in_uops)
    if x.op is UnaryOps.BITCAST: return UOp(UOps.BITCAST, x.arg.scalar(), in_uops)
    if x.op in ReduceOps:
      op = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[cast(ReduceOps, x.op)]
      # NOTE: always using ridxs is fine here
      return UOp(UOps.REDUCE, x.dtype, (in_uops[0], UOp.const(x.dtype, get_reduce_acc(x))) + tuple(self.ridxs[i] for i in x.arg), op)
    return UOp.alu(x.op, *in_uops)

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  def linearize(self) -> Lowerer:
    self.uop_cache: Dict[LazyOp, UOp] = {}

    # kernel name (before late upcast)
    self.name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.lazyops) else "E")) + \
                 (f"{len(self.outbufs)}_" if len(self.outbufs) > 1 else "_") + \
                 colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])
    if DEBUG >= 4: print(self.name)

    # name the function something unique
    Lowerer.kernel_cnt[(function_name := to_function_name(self.name))] += 1
    suffix = f"{'n'+str(Lowerer.kernel_cnt[function_name]-1)}" if Lowerer.kernel_cnt[function_name] > 1 else ""
    self.name = self.name+colored(suffix, 'BLACK')

    self.idxs = []
    # add a local buffer for multistage reduce.
    if self.group_for_reduces:
      for i in range(len(self.reduceops)):
        # TODO: the strides of this can be controlled
        self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+self.group_for_reduces]) + [1] * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        temp_dtype = cast(LazyOp, self.reduceop).dtype
        self.bufs.append(LocalBuffer(name:=f"temp{i if len(self.reduceops) > 1 else ''}", buf_size:=self.sts[-1].size, temp_dtype))
        self.local_buffer_uop = UOp(UOps.DEFINE_LOCAL, PtrDType(temp_dtype), (), (name, buf_size))

    #from tinygrad.engine.graph import print_tree
    #print_tree(self.ast[0])

    # set the shapetrackers to the optimized ones, fixup reduceop
    # transformed to the final LazyOp
    @functools.lru_cache(None)
    def fixup_ast(op:LazyOp) -> LazyOp:
      if op.op in BufferOps:
        arg = replace(op.arg, st=self.sts[self.bufs.index(op.arg)])
      elif op.op in ReduceOps:
        arg = tuple(i for i in range(self.first_reduce+self.group_for_reduces, self.shape_len) if self.full_shape[i] != self.sts[0].shape[i])
        if self.group_for_reduces:
          start = LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), arg)
          local_buffer = MemBuffer(-1, start.dtype, self.sts[-1])
          local_store = LazyOp(BufferOps.STORE, (start,), local_buffer)
          local_load = LazyOp(BufferOps.LOAD, (local_store,), local_buffer)
          return LazyOp(op.op, (local_load,), tuple(range(self.first_reduce, self.first_reduce+self.group_for_reduces)))
      else:
        arg = op.arg
      return LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), arg)
    modified_ast = tuple(fixup_ast(x) for x in self.ast)

    if self.opts.has_local:
      # define indexes
      global_idxs, loop_global_idxs = get_grouped_dims("gidx", 0, self.full_shape[:self.global_dims], 3 if self.opts.has_local else 0)
      local_idxs, loop_local_idxs = get_grouped_dims("lidx", self.global_dims, self.full_shape[self.global_dims:self.first_reduce+self.group_for_reduces], 3 if self.opts.has_local else 0)  # noqa: E501
      self.idxs = global_idxs + local_idxs

      # define sizes
      self.global_size: Optional[List[int]] = [x.arg[2] for x in loop_global_idxs]
      self.local_size: Optional[List[int]] = [x.arg[2] for x in loop_local_idxs]
      self.global_size += [1]*(3-len(self.global_size))
      self.local_size += [1]*(3-len(self.local_size))
    else:
      # all loops
      self.idxs = []
      for i,g in enumerate(self.full_shape[:self.first_reduce]):
        self.idxs.append(UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), variable_to_uop(g)), (i, False, False)))
      self.global_size, self.local_size = None, None

    # reduce loops
    for i,g in enumerate(self.full_shape[self.first_reduce+self.group_for_reduces:], start=self.first_reduce+self.group_for_reduces):
      unrolled, is_reduce = i >= (self.shape_len-self.upcasted), self.full_shape[i] != self.output_shape[i]
      self.idxs.append(UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), variable_to_uop(g)), (i, unrolled, is_reduce)))

    # late indexes
    self.ridxs = self.idxs[:]
    for a in range(self.first_reduce, self.first_reduce+self.group_for_reduces):
      self.ridxs[a] = UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), variable_to_uop(self.full_shape[a])), (1000+a, False, True))

    self.uops:UOpGraph = UOpGraph([self.to_uop(x) for x in modified_ast])

    # maybe graph the uops
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS"):
      self.uops.graph()
      if getenv("GRAPHUOPS") == 2: exit(0)
    return self

  def to_program(self) -> Program:
    self.linearize()
    src = self.opts.render(to_function_name(self.name), self.uops)
    info = get_lazyop_info(self.ast[0])
    ops, mem = self.uops.flops_mem()
    run_count = prod((self.global_size if self.global_size else []) + (self.local_size if self.local_size else []))
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size,
                   self.uops, min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count))
