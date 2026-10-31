#!/usr/bin/env python3
"""PTX (nvcc -O3 sm_90, libdevice inlined) -> C translator for cuda_math.h.
Usage: gen_cuda_math.py ref.ptx fast.ptx sass/kernels/cuda_math.h
  ref.ptx:  nvcc -ptx -arch=sm_90 -O3 sass/tools/cuda_math_ref.cu
  fast.ptx: nvcc -ptx -arch=sm_90 -O3 -use_fast_math -ftz=false sass/tools/cuda_math_ref.cu
Rewrites the generated section of cuda_math.h
Each `k_<name>` kernel is translated op-for-op into a C function body using the
__sm_* building-block macros defined in cuda_math.h."""
import re, struct, sys

# public C signatures: name -> (ret, [param decls], [output templates]) ; outputs index by store slot
# a<i> = i-th loaded input, outputs: template with {} = value expression
F, I, U, L, UL = 'float', 'int', 'unsigned', 'long long', 'unsigned long long'
SPEC = {}
def unary(names, rt=F, at=F):
    for n in names: SPEC[n] = (rt, [f'{at} a0'], ['return {};'])
def binary(names):
    for n in names: SPEC[n] = (F, ['float a0', 'float a1'], ['return {};'])
def ternary(names):
    for n in names: SPEC[n] = (F, ['float a0', 'float a1', 'float a2'], ['return {};'])
unary('expf exp2f exp10f expm1f logf log2f log10f log1pf sqrtf rsqrtf cbrtf sinf cosf tanf tanhf erff erfcf atanf '
      'asinf acosf sinhf coshf floorf ceilf truncf roundf rintf nearbyintf asinhf acoshf atanhf erfinvf erfcinvf '
      'lgammaf tgammaf logbf sinpif cospif rcbrtf normcdff normcdfinvf erfcxf __expf __exp10f __logf __log2f __log10f '
      '__sinf __cosf __tanf __frcp_rn __frcp_rz __frcp_ru __frcp_rd __fsqrt_rn __fsqrt_rz __fsqrt_ru __fsqrt_rd '
      '__frsqrt_rn __saturatef'.split())
binary('powf atan2f fmodf remainderf copysignf fdimf hypotf __powf __fdividef'.split())
binary([f'__f{o}_{r}' for o in ('div', 'add', 'sub', 'mul') for r in ('rn', 'rz', 'ru', 'rd')])
ternary(['fmaf'] + [f'__fmaf_{r}' for r in ('rn', 'rz', 'ru', 'rd')])
unary(['ilogbf', '__isnanf', '__isinff', '__finitef', '__signbitf'] + [f'__float2int_{r}' for r in ('rn', 'rz', 'ru', 'rd')], rt=I)
for n in ('isnan', 'isinf', 'isfinite', 'signbit'): SPEC[n] = ('bool', ['float a0'], ['return {} != 0;'])
unary([f'__float2uint_{r}' for r in ('rn', 'rz', 'ru', 'rd')], rt=U)
unary([f'__float2ll_{r}' for r in ('rn', 'rz', 'ru', 'rd')], rt=L)
unary([f'__float2ull_{r}' for r in ('rn', 'rz', 'ru', 'rd')], rt=UL)
for p, t in (('int', I), ('uint', U), ('ll', L), ('ull', UL)):
    unary([f'__{p}2float_{r}' for r in ('rn', 'rz', 'ru', 'rd')], at=t)
SPEC['sincosf'] = ('void', ['float a0', 'float *o0', 'float *o1'], ['*o0 = {};', '*o1 = {};'])
SPEC['__sincosf'] = SPEC['sincosf']
SPEC['sincospif'] = SPEC['sincosf']
SPEC['ldexpf'] = (F, ['float a0', 'int a1'], ['return {};'])
SPEC['scalbnf'] = SPEC['ldexpf']
SPEC['frexpf'] = (F, ['float a0', 'int *o1'], ['return {};', '*o1 = __float_as_int({});'])
SPEC['modff'] = (F, ['float a0', 'float *o1'], ['return {};', '*o1 = {};'])
SPEC['remquof'] = (F, ['float a0', 'float a1', 'int *o2'], ['return {};', None, '*o2 = __float_as_int({});'])

CT = {'f32': 'float', 'f64': 'double', 's32': 'int', 'u32': 'unsigned', 'b32': 'unsigned', 's64': 'long long',
      'u64': 'unsigned long long', 'b64': 'unsigned long long', 'pred': 'bool', 's16': 'short', 'u16': 'unsigned short'}
RCLASS = {'f': 'float', 'fd': 'double', 'r': 'unsigned', 'rd': 'unsigned long long', 'p': 'bool', 'rs': 'unsigned short'}
RND = {'rn': 'rn', 'rz': 'rz', 'rm': 'rm', 'rp': 'rp'}


def fimm(tok):
    bits = int(tok[2:], 16)
    if (bits & 0x7f800000) == 0x7f800000:
        return f'__sm_f32bits(0x{bits:08x}u)'
    v = struct.unpack('<f', struct.pack('<I', bits))[0]
    if v == 0:
        return '-0.0f' if bits >> 31 else '0.0f'
    m, e = v.hex().split('p')
    m = m.rstrip('0').rstrip('.')
    return f'{m}p{e}f'


def dimm(tok):
    v = struct.unpack('<d', struct.pack('<Q', int(tok[2:], 16)))[0]
    return v.hex()


def reg_class(r):
    m = re.match(r'%([a-z]+)\d+$', r)
    return m.group(1) if m else None


class Fn:
    def __init__(s, name, lines):
        s.name, s.lines = name, lines
        s.addr = set()
        s.depot = 0
        s.inputs = {}  # slot -> reg
        s.outputs = {}  # slot -> expr
        s.regs = set()
        s.out = []
        s.labels_used = set()

    def v(s, tok, ty):
        """operand tok read as C type ty"""
        tok = tok.strip()
        ct = CT[ty]
        if tok.startswith('%'):
            if tok in ('%SPL', '%SP'):
                return '0ull'
            rc = reg_class(tok)
            s.regs.add(tok)
            name = tok[1:]
            if rc == 'p':
                return name
            if RCLASS[rc] == ct:
                return name
            if rc in ('f', 'fd'):
                raise ValueError(f'float reg {tok} as {ty}')
            if ty in ('f32', 'f64'):
                raise ValueError(f'int reg {tok} as {ty}')
            return f'({ct}){name}'
        if tok.startswith('0f'):
            return fimm(tok)
        if tok.startswith('0d'):
            return dimm(tok)
        n = int(tok, 0)
        bits = 64 if ty.endswith('64') else 32
        n &= (1 << bits) - 1
        suf = 'ull' if bits == 64 else 'u'
        lit = f'0x{n:x}{suf}' if n > 9 else f'{n}{suf}'
        if ty.startswith('s'):
            return f'({ct}){lit}'
        if ty.startswith('f'):
            return f'({ct}){int(tok, 0)}'
        return lit

    def d(s, tok):
        s.regs.add(tok)
        return tok[1:]

    def emit(s, line, pred=None):
        if pred:
            neg = pred.startswith('!')
            p = pred.lstrip('!')[1:]
            line = f'if ({"!" if neg else ""}{p}) {line}'
        s.out.append('  ' + line)

    def translate(s):
        body = []
        for ln in s.lines:
            ln = ln.split('//')[0].strip()
            if not ln or ln.startswith('.reg') or ln.startswith('.pragma') or ln in ('{', '}'):
                continue
            if ln.startswith('.local'):
                s.depot = int(re.search(r'\[(\d+)\]', ln).group(1))
                continue
            body.append(ln)
        for ln in body:
            if ln.endswith(':'):
                s.out.append(f'{ln[:-1].replace("$L__BB", "L")}:;')
                continue
            ln = ln.rstrip(';')
            pred = None
            if ln.startswith('@'):
                pred, ln = ln.split(None, 1)
                pred = pred[1:]
            op, _, rest = ln.partition(' ')
            ops = [o.strip() for o in re.split(r',(?![^\[]*\])', rest.strip())] if rest.strip() else []
            s.inst(op, ops, pred)
        return s.out

    def mem(s, tok):
        m = re.match(r'\[(%?[\w$]+)(?:\+(-?\d+))?\]', tok)
        return m.group(1), int(m.group(2) or 0)

    def inst(s, op, a, pred):
        p = op.split('.')
        base = p[0]
        ty = p[-1]
        mods = p[1:-1]
        A = s.addr
        srcregs = [x for x in a[1:] if x.startswith('%')]
        # ---- address plumbing of the test kernel (dropped) ----
        if base == 'ld' and 'param' in mods:
            A.add(a[0]); return
        if base == 'cvta' or (base == 'mov' and a[1].startswith('%tid')):
            A.add(a[0]); return
        if base in ('mul', 'add', 'shl', 'cvt') and any(x in A for x in srcregs) and '%SPL' not in a:
            A.add(a[0]); return
        if base == 'ret':
            s.emit('return __ret;' if SPEC[s.name][0] != 'void' else 'return;', pred); return
        if base == 'mov' and ty == 'u64' and a[1] == '%SPL' or base == 'mov' and a[1].startswith('__local_depot'):
            return
        if base == 'mov' and a[1] == '__cudart_i2opi_f':
            s.emit(f'{s.d(a[0])} = 0ull;'); return
        if base == 'ld' and 'global' in mods and 'nc' not in mods:
            r, off = s.mem(a[1]); assert r in A
            slot = off // (4 if ty.endswith('32') else 8) // 32
            s.inputs[slot] = (a[0], ty); s.regs.add(a[0]); return
        if base == 'st' and 'global' in mods:
            r, off = s.mem(a[0]); assert r in A
            slot = off // (4 if ty.endswith('32') else 8) // 32
            spec = SPEC[s.name][2]
            t = spec[slot]
            if t == 'return {};': t = '__ret = {};'
            elif t.startswith('return '): t = '__ret = ' + t[7:]
            s.emit(t.format(s.v(a[1], ty)), pred); return
        if base == 'ld' and 'nc' in mods:
            r, off = s.mem(a[1])
            s.emit(f'{s.d(a[0])} = __sm_i2opi_f[({s.v(r, "u64")}{off:+d}) >> 2];'.replace('+0)', ')'), pred); return
        if base == 'ld' and 'local' in mods:
            r, off = s.mem(a[1])
            s.emit(f'{s.d(a[0])} = __d[({s.v(r, "u64")}{off:+d}) >> 2];'.replace('+0)', ')'), pred); return
        if base == 'st' and 'local' in mods:
            r, off = s.mem(a[0])
            s.emit(f'__d[({s.v(r, "u64")}{off:+d}) >> 2] = (unsigned){s.v(a[1], "u64")};'.replace('+0)', ')'), pred); return
        if base == 'bra':
            lab = a[0].replace('$L__BB', 'L')
            s.emit(f'goto {lab};', pred); return
        D = s.d(a[0]) if a else None
        V = lambda i, t=ty: s.v(a[i], t)
        rnd = next((m for m in mods if m in RND), None)
        ftz = 'ftz' in mods
        fz = '_ftz' if ftz else ''
        e = None
        if ty == 'f32' or ty == 'f64':
            if base == 'mov': e = V(1)
            elif base == 'fma':
                e = f'__sm_fma_{rnd}{fz}({V(1)}, {V(2)}, {V(3)})'
            elif base in ('add', 'sub', 'mul'):
                if rnd:
                    y = V(2) if base != 'sub' else f'-({V(2)})' if rnd == 'rn' or True else None
                    e = f'__sm_{"add" if base != "mul" else "mul"}_{rnd}{fz}({V(1)}, {y})'
                else:
                    assert not ftz or ty == 'f32', op
                    o = {'add': '+', 'sub': '-', 'mul': '*'}[base]
                    e = f'{V(1)} {o} {V(2)}'
                    if ftz: e = f'__sm_ftz({e})'
            elif base == 'neg': e = f'-{V(1)}'
            elif base == 'abs': e = f'__sm_fabs({V(1)})'
            elif base == 'min': e = f'__sm_fmin{fz}({V(1)}, {V(2)})'
            elif base == 'max': e = f'__sm_fmax{fz}({V(1)}, {V(2)})'
            elif base == 'copysign': e = f'__sm_copysign({V(2)}, {V(1)})'
            elif base in ('ex2', 'lg2', 'rcp', 'rsqrt', 'sqrt', 'sin', 'cos', 'tanh'):
                e = f'__sm_{base}_{"approx" if "approx" in mods else rnd}{fz}({V(1)})'
            elif base == 'div':
                e = f'__sm_div_{"approx" if "approx" in mods else ("full" if "full" in mods else rnd)}{fz}({V(1)}, {V(2)})'
            elif base == 'selp': e = f'{V(3, "pred")} ? {V(1)} : {V(2)}'
            elif base == 'testp':
                pass
        if e is None and base == 'setp':
            cmp, t = mods[0], ty
            x, y = V(1, t), V(2, t)
            if t in ('f32', 'f64'):
                ordered = {'eq': '==', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}
                if cmp in ordered: e = f'{x} {ordered[cmp]} {y}'
                elif cmp == 'ne': e = f'__sm_one({x}, {y})'
                elif cmp == 'neu': e = f'{x} != {y}'
                elif cmp == 'equ': e = f'!__sm_one({x}, {y})'
                elif cmp.endswith('u') and cmp[:-1] in ordered:
                    inv = {'lt': '>=', 'le': '>', 'gt': '<=', 'ge': '<'}[cmp[:-1]]
                    e = f'!({x} {inv} {y})'
                elif cmp == 'num': e = f'!__builtin_isnan({x}) && !__builtin_isnan({y})'
                elif cmp == 'nan': e = f'__builtin_isnan({x}) || __builtin_isnan({y})'
            else:
                c = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'lo': '<', 'ls': '<=', 'hi': '>', 'hs': '>='}[cmp]
                e = f'{x} {c} {y}'
            # f32 setp with ftz: denormals compare as zero
            if 'ftz' in mods: e = e.replace(x, f'__sm_ftz({x})').replace(y, f'__sm_ftz({y})')
        if e is None and ty == 'pred':
            if base == 'and': e = f'{V(1)} && {V(2)}'
            elif base == 'or': e = f'{V(1)} || {V(2)}'
            elif base == 'not': e = f'!{V(1)}'
            elif base == 'xor': e = f'{V(1)} != {V(2)}'
            elif base == 'mov': e = V(1)
        if e is None and base == 'cvt':
            dt, st = mods[-1], ty
            m = [x for x in mods[:-1] if x not in ('ftz',)]
            r = m[0] if m else None
            sat = 'sat' in mods
            if dt == 'f32' and st == 'f32':
                if sat: e = f'__sm_saturate{fz}({V(1)})'
                else: e = {'rzi': '__sm_trunc', 'rni': '__sm_rint', 'rmi': '__sm_floor', 'rpi': '__sm_ceil'}[r] + f'({V(1)})'
                if ftz and not sat: e = f'__sm_ftz({e})'
            elif st == 'f32' and dt in ('s32', 'u32', 's64', 'u64'):
                k = {'s32': 'i', 'u32': 'ui', 's64': 'll', 'u64': 'ull'}[dt]
                e = f'__sm_f2{k}_{r[:2]}{fz}({V(1)})'
            elif dt == 'f32' and st in ('s32', 'u32', 's64', 'u64'):
                k = {'s32': 'i', 'u32': 'ui', 's64': 'll', 'u64': 'ull'}[st]
                e = f'__sm_{k}2f_{r}({V(1)})'
            elif dt == 'f64' and st == 's64': assert r == 'rn'; e = f'(double){V(1)}'
            elif dt == 'f32' and st == 'f64': assert r == 'rn'; e = f'(float){V(1)}'
            elif dt in ('u64', 's64') and st in ('u32', 's32'): e = f'({CT[dt]}){V(1)}'
            elif dt in ('u32', 's32') and st in ('u64', 's64'): e = f'({CT[dt]}){V(1)}'
            elif dt in ('u32', 's32') and st in ('u16', 's16', 'u8', 's8'): e = f'({CT[dt]})({ {"u16":"unsigned short","s16":"short","u8":"unsigned char","s8":"signed char"}[st]}){V(1, "u32")}'
        if e is None and ty in CT and ty[0] in 'bsu':
            w = 64 if ty.endswith('64') else 32
            ut = 'u64' if w == 64 else 'u32'
            if base == 'mov':
                rc = reg_class(a[1]) if a[1].startswith('%') else None
                if rc == 'f': e = f'__float_as_uint({a[1][1:]})'; s.regs.add(a[1])
                elif rc == 'fd': e = f'__builtin_bit_cast(unsigned long long, {a[1][1:]})'; s.regs.add(a[1])
                elif reg_class(a[0]) == 'f': e = f'__sm_f32bits({V(1, "u32")})'
                elif reg_class(a[0]) == 'fd': e = f'__builtin_bit_cast(double, {V(1, "u64")})'
                else: e = V(1)
            elif base in ('and', 'or', 'xor'):
                e = f'{V(1)} {dict(zip(("and","or","xor"),"&|^"))[base]} {V(2)}'
            elif base == 'not': e = f'~{V(1)}'
            elif base == 'shl': e = f'{V(1)} << {V(2, "u32")}'
            elif base == 'shr': e = f'{V(1)} >> {V(2, "u32")}'
            elif base == 'add': e = f'{V(1, ut)} + {V(2, ut)}'
            elif base == 'sub': e = f'{V(1, ut)} - {V(2, ut)}'
            elif base == 'neg': e = f'-{V(1, ut)}'
            elif base == 'abs': e = f'{V(1)} < 0 ? -{V(1)} : {V(1)}'
            elif base == 'min': e = f'{V(1)} < {V(2)} ? {V(1)} : {V(2)}'
            elif base == 'max': e = f'{V(1)} > {V(2)} ? {V(1)} : {V(2)}'
            elif base == 'mul' and mods == ['lo']: e = f'{V(1, ut)} * {V(2, ut)}'
            elif base == 'mul' and mods == ['hi']: e = f'__sm_mulhi_{ty}({V(1)}, {V(2)})'
            elif base == 'mad' and mods == ['lo']: e = f'{V(1, ut)} * {V(2, ut)} + {V(3, ut)}'
            elif base == 'mul' and mods == ['wide']:
                wt = {'u32': 'unsigned long long', 's32': 'long long'}[ty]
                e = f'({wt}){V(1)} * ({wt}){V(2)}'
            elif base == 'mad' and mods == ['wide']:
                wt = {'u32': 'unsigned long long', 's32': 'long long'}[ty]
                e = f'({wt}){V(1)} * ({wt}){V(2)} + {s.v(a[3], ty[0] + "64")}'
            elif base == 'selp': e = f'{V(3, "pred")} ? {V(1)} : {V(2)}'
            elif base == 'clz': e = f'__sm_clz{w}({V(1)})'
            elif base == 'popc': e = f'__sm_popc{w}({V(1)})'
            elif base == 'brev': e = f'__sm_brev{w}({V(1)})'
            elif base == 'bfi':
                # bfi.b64 d, a, b, pos, len  (insert a into b)
                e = f'__sm_bfi{w}({V(1)}, {V(2)}, {V(3, "u32")}, {V(4, "u32")})'
            elif base == 'bfe': e = f'__sm_bfe_{ty}({V(1)}, {V(2, "u32")}, {V(3, "u32")})'
            elif base == 'prmt': e = f'__sm_prmt({V(1)}, {V(2)}, {V(3)})'
            elif base == 'mad24' or base == 'mul24': pass
        if e is None:
            raise ValueError(f'{s.name}: unhandled {op} {a}')
        s.emit(f'{D} = {e};', pred)

    def render(s):
        rt, params, outs = SPEC[s.name]
        body = s.translate()
        # inputs
        decl = {}
        for r in sorted(s.regs, key=lambda r: (reg_class(r), int(re.sub(r'\D', '', r)))):
            decl.setdefault(RCLASS[reg_class(r)], []).append(r[1:])
        pre = []
        for t, rs in decl.items():
            for i in range(0, len(rs), 12):
                pre.append(f'  {t} {", ".join(rs[i:i + 12])};')
        if s.depot:
            pre.append(f'  unsigned __d[{s.depot // 4}];')
        for slot, (r, ty) in sorted(s.inputs.items()):
            src = f'a{slot}'
            rc = reg_class(r)
            pt = SPEC[s.name][1][slot].rsplit(' ', 1)[0]
            if rc == 'f' and pt != 'float': src = f'__sm_f32bits(a{slot})'
            elif rc == 'r' and pt == 'float': src = f'__float_as_uint(a{slot})'
            elif rc == 'rd' and pt == 'float': raise ValueError(s.name)
            pre.append(f'  {r[1:]} = {src};')
        if rt != 'void':
            pre.insert(0, f'  {rt} __ret;')
        return f'SASS_MATH_FN({rt}, {s.name})({", ".join(params)}) {{\n' + '\n'.join(pre + body) + '\n}\n'


# tanhf's fast-math body uses tanh.approx.f32, which has no clang builtin: it stays precise.
NO_FAST = {'tanhf'}
BEGIN, END = '// ---- BEGIN GENERATED (gen_cuda_math.py) ----\n', '// ---- END GENERATED ----\n'


def kernels(path):
    txt = open(path).read()
    return {m.group(1): m.group(2).split('\n')
            for m in re.finditer(r'\.entry k_(\w+)\(.*?\n\)\n\{(.*?)\n\}', txt, re.S)}


def main():
    ref, fast, hdr = kernels(sys.argv[1]), kernels(sys.argv[2]), sys.argv[3]
    norm = lambda ls: [re.sub(r'BB\d+_', 'BB_', l) for l in ls]
    out = []
    for n in sorted(SPEC, key=lambda n: not n.startswith('__')):
        body = Fn(n, ref[n]).render()
        if n not in NO_FAST and norm(ref[n]) != norm(fast[n]):
            body = f'#if SASS_MATH_FAST\n{Fn(n, fast[n]).render()}#else\n{body}#endif\n'
        out.append(body)
    h = open(hdr).read()
    a, b = h.index(BEGIN) + len(BEGIN), h.index(END)
    open(hdr, 'w').write(h[:a] + '\n'.join(out) + h[b:])


if __name__ == '__main__':
    main()
