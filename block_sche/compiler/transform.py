from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


@dataclass(frozen=True)
class TransformOptions:
    kernel_name: Optional[str] = None
    output_kernel_name: Optional[str] = None
    schedule_param: str = "__bs_schedule"
    task_param: str = "__bs_task"
    include_runtime: bool = True
    sm_affine: bool = False
    trace: bool = False
    host_launcher: bool = False
    keep_original: bool = True


@dataclass(frozen=True)
class Kernel:
    prefix_start: int
    signature_start: int
    params_start: int
    params_end: int
    body_start: int
    body_end: int
    template_prefix: str
    qualifier_prefix: str
    name: str
    params: str
    body: str


class TransformError(ValueError):
    pass


def transform_cuda_source(source: str, options: TransformOptions = TransformOptions()) -> str:
    kernel = _find_kernel(source, options.kernel_name)
    generated = _emit_kernel_artifacts(kernel, options)

    pieces: List[str] = []
    if options.include_runtime and "#include <block_sche/block_sche.cuh>" not in source:
        pieces.append("#include <block_sche/block_sche.cuh>\n\n")

    if options.keep_original:
        pieces.append(source.rstrip())
        pieces.append("\n\n")
    else:
        pieces.append(source[: kernel.prefix_start].rstrip())
        pieces.append("\n\n")
        pieces.append(source[kernel.body_end + 1 :].lstrip())
        if pieces[-1] and not pieces[-1].endswith("\n"):
            pieces[-1] += "\n"

    pieces.append(generated)
    return "".join(pieces)


def transform_all_cuda_kernels(source: str, options: TransformOptions = TransformOptions()) -> str:
    if options.kernel_name is not None:
        raise TransformError("--all cannot be combined with a specific kernel name")
    if options.output_kernel_name is not None:
        raise TransformError("--all cannot be combined with --output-kernel")

    kernels = _find_kernels(source, None)
    if not kernels:
        raise TransformError("could not find a __global__ kernel")

    pieces: List[str] = []
    if options.include_runtime and "#include <block_sche/block_sche.cuh>" not in source:
        pieces.append("#include <block_sche/block_sche.cuh>\n\n")
    if options.keep_original:
        pieces.append(source.rstrip())
        pieces.append("\n\n")
    else:
        pieces.append(_drop_kernel_definitions(source, kernels).rstrip())
        pieces.append("\n\n")

    pieces.append("\n\n".join(_emit_kernel_artifacts(kernel, options) for kernel in kernels))
    return "".join(pieces)


def _find_kernel(source: str, kernel_name: Optional[str]) -> Kernel:
    matches = _find_kernels(source, kernel_name)
    if not matches:
        target = f" named {kernel_name!r}" if kernel_name else ""
        raise TransformError(f"could not find a __global__ kernel{target}")
    if len(matches) > 1 and kernel_name is None:
        names = ", ".join(kernel.name for kernel in matches)
        raise TransformError(f"multiple __global__ kernels found ({names}); pass --kernel")
    return matches[0]


def _find_kernels(source: str, kernel_name: Optional[str]) -> List[Kernel]:
    code_mask = _code_position_mask(source)
    matches = []
    for global_match in re.finditer(r"\b__global__\b", source):
        global_pos = global_match.start()
        if not code_mask[global_pos]:
            continue

        prefix_start, template_prefix, qualifier_start = _kernel_prefix(source, global_pos)
        parsed = _parse_kernel_signature_after_global(source, global_match.end())
        if parsed is None:
            continue
        name, params_start = parsed
        if kernel_name is not None and name != kernel_name:
            continue
        params_end = _find_matching(source, params_start, "(", ")")
        body_start = _skip_ws(source, params_end + 1)
        if body_start >= len(source) or source[body_start] != "{":
            continue
        body_end = _find_matching(source, body_start, "{", "}")
        matches.append(
            Kernel(
                prefix_start=prefix_start,
                signature_start=qualifier_start,
                params_start=params_start,
                params_end=params_end,
                body_start=body_start,
                body_end=body_end,
                template_prefix=template_prefix,
                qualifier_prefix=source[qualifier_start:params_start],
                name=name,
                params=source[params_start + 1 : params_end],
                body=source[body_start + 1 : body_end],
            )
        )
    return matches


def _kernel_prefix(source: str, global_pos: int) -> tuple[int, str, int]:
    line_start = source.rfind("\n", 0, global_pos) + 1
    qualifier_start = _find_leading_launch_bounds_start(source, global_pos)
    if qualifier_start is None:
        qualifier_start = _skip_inline_ws_forward(source, line_start)
    prefix_start = qualifier_start

    template_range = _find_template_prefix_range(source, qualifier_start)
    if template_range is not None:
        template_start, template_end = template_range
        prefix_start = template_start
        return prefix_start, source[template_start:template_end], qualifier_start

    extern_start = _find_extern_c_prefix_start(source, qualifier_start)
    if extern_start is not None:
        prefix_start = extern_start
        return prefix_start, source[prefix_start:qualifier_start], qualifier_start

    return prefix_start, "", qualifier_start


def _find_leading_launch_bounds_start(source: str, global_pos: int) -> Optional[int]:
    before_global = _skip_ws_backward(source, global_pos)
    attr_start = source.rfind("__launch_bounds__", 0, before_global)
    if attr_start == -1:
        return None
    call_start = _skip_ws(source, attr_start + len("__launch_bounds__"))
    if call_start >= len(source) or source[call_start] != "(":
        return None
    call_end = _find_matching(source, call_start, "(", ")")
    if _skip_ws(source, call_end + 1) == global_pos:
        return attr_start
    return None


def _find_template_prefix_range(source: str, qualifier_start: int) -> Optional[tuple[int, int]]:
    i = qualifier_start
    while True:
        i = _skip_ws_backward(source, i)
        start = source.rfind("template", 0, i)
        if start == -1:
            return None
        if not _is_word_at(source, start, "template"):
            i = start
            continue
        lt = _skip_ws(source, start + len("template"))
        if lt < qualifier_start and source[lt] == "<":
            end = _find_matching(source, lt, "<", ">")
            prefix_end = _skip_ws(source, end + 1)
            if end < qualifier_start and not source[prefix_end:qualifier_start].strip():
                return start, prefix_end
        i = start


def _find_extern_c_prefix_start(source: str, qualifier_start: int) -> Optional[int]:
    line_start = source.rfind("\n", 0, qualifier_start) + 1
    candidate = _skip_inline_ws_forward(source, line_start)
    if source.startswith('extern "C"', candidate):
        return candidate
    return None


def _parse_kernel_signature_after_global(source: str, start: int) -> Optional[tuple[str, int]]:
    i = start
    while i < len(source):
        i = _skip_ws(source, i)
        if source.startswith("__launch_bounds__", i):
            i = _skip_attribute_call(source, i + len("__launch_bounds__"))
            continue
        if source.startswith("__host__", i) or source.startswith("__device__", i):
            i = _skip_identifier(source, i)
            continue
        if source.startswith("void", i) and _is_word_at(source, i, "void"):
            i += len("void")
            continue
        name_match = re.match(_IDENT, source[i:])
        if name_match is None:
            return None
        name = name_match.group(0)
        if name == "__launch_bounds__":
            i = _skip_attribute_call(source, i + len(name))
            continue
        i += len(name)
        i = _skip_ws(source, i)
        if i < len(source) and source[i] == "(":
            return name, i
        return None
    return None


def _skip_attribute_call(source: str, start: int) -> int:
    i = _skip_ws(source, start)
    if i < len(source) and source[i] == "(":
        return _find_matching(source, i, "(", ")") + 1
    return i


def _skip_identifier(source: str, start: int) -> int:
    match = re.match(_IDENT, source[start:])
    return start + len(match.group(0)) if match else start


def _is_word_at(source: str, start: int, word: str) -> bool:
    if not source.startswith(word, start):
        return False
    before = source[start - 1] if start > 0 else ""
    after_pos = start + len(word)
    after = source[after_pos] if after_pos < len(source) else ""
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _skip_inline_ws_forward(source: str, start: int) -> int:
    i = start
    while i < len(source) and source[i] in " \t\r":
        i += 1
    return i


def _skip_ws_backward(source: str, end: int) -> int:
    i = end
    while i > 0 and source[i - 1].isspace():
        i -= 1
    return i


def _drop_kernel_definitions(source: str, kernels: Sequence[Kernel]) -> str:
    out: List[str] = []
    cursor = 0
    for kernel in sorted(kernels, key=lambda item: item.prefix_start):
        out.append(source[cursor : kernel.prefix_start])
        cursor = kernel.body_end + 1
    out.append(source[cursor:])
    return "".join(out)


def _emit_kernel_artifacts(kernel: Kernel, options: TransformOptions) -> str:
    _validate_kernel_body(kernel.body, kernel.name)
    persistent_name = options.output_kernel_name or f"{kernel.name}_persistent"
    device_name = f"{kernel.name}_rtask"
    launch_bounds = _kernel_launch_bounds(kernel)
    launch_bounds_suffix = f" {launch_bounds}" if launch_bounds else ""

    params = _split_params(kernel.params)
    param_names = [_param_name(param) for param in params if param.strip() and param.strip() != "void"]
    call_args = ", ".join([options.task_param, options.schedule_param, *param_names])

    device_params = _join_params(
        [
            f"block_sche::RTask {options.task_param}",
            f"block_sche::DeviceSchedule {options.schedule_param}",
            *params,
        ]
    )
    persistent_extra_params = ["block_sche::DeviceSchedule " + options.schedule_param]
    if options.trace:
        persistent_extra_params.append("block_sche::DeviceTrace __bs_trace")
    persistent_params = _join_params([*persistent_extra_params, *params])

    rewritten_body = _wrap_with_virtual_cuda_indices(
        kernel.body, options.task_param, options.schedule_param
    )
    device_function = (
        f"{kernel.template_prefix}"
        f"__device__ __forceinline__ void {device_name}({device_params}) "
        "{\n"
        f"{rewritten_body}\n"
        "}\n"
    )

    if options.sm_affine:
        persistent_body = (
            "  const uint32_t __bs_sm = block_sche::current_sm_id();\n"
            f"  if (__bs_sm >= {options.schedule_param}.sm_count || "
            f"{options.schedule_param}.sm_offsets == nullptr || "
            f"{options.schedule_param}.sm_task_ids == nullptr || "
            f"{options.schedule_param}.sm_cursors == nullptr) {{\n"
            "    return;\n"
            "  }\n"
            f"  const uint32_t __bs_begin = {options.schedule_param}.sm_offsets[__bs_sm];\n"
            f"  const uint32_t __bs_end = {options.schedule_param}.sm_offsets[__bs_sm + 1];\n"
            "  __shared__ uint32_t __bs_claimed_pos;\n"
            "  __shared__ uint32_t __bs_task_block_x;\n"
            "  __shared__ uint32_t __bs_task_block_y;\n"
            "  __shared__ uint32_t __bs_task_block_z;\n"
            "  __shared__ uint32_t __bs_task_linear_block;\n"
            "  __shared__ uint32_t __bs_task_sm;\n"
            "  while (true) {\n"
            "    if (block_sche::is_block_leader()) {\n"
            f"      const uint32_t __bs_local = atomicAdd(&{options.schedule_param}.sm_cursors[__bs_sm], 1);\n"
            "      __bs_claimed_pos = __bs_begin + __bs_local;\n"
            "      if (__bs_claimed_pos < __bs_end) {\n"
            f"        const block_sche::RTask __bs_loaded_task = "
            f"{options.schedule_param}.tasks[{options.schedule_param}.sm_task_ids[__bs_claimed_pos]];\n"
            "        __bs_task_block_x = __bs_loaded_task.block.x;\n"
            "        __bs_task_block_y = __bs_loaded_task.block.y;\n"
            "        __bs_task_block_z = __bs_loaded_task.block.z;\n"
            "        __bs_task_linear_block = __bs_loaded_task.linear_block;\n"
            "        __bs_task_sm = __bs_loaded_task.sm;\n"
            "      }\n"
            "    }\n"
            "    __syncthreads();\n"
            "    if (__bs_claimed_pos >= __bs_end) {\n"
            "      break;\n"
            "    }\n"
            f"    const block_sche::RTask {options.task_param} = "
            "{block_sche::Dim3u(__bs_task_block_x, __bs_task_block_y, __bs_task_block_z), "
            "__bs_task_linear_block, __bs_task_sm};\n"
            f"{_trace_statement(options.trace, options.task_param, '__bs_sm')}"
            f"    {device_name}({call_args});\n"
            "    __syncthreads();\n"
            "  }\n"
        )
    else:
        persistent_body = (
            "  const uint32_t __bs_cta = block_sche::persistent_cta_id();\n"
            f"  const uint32_t __bs_stride = {options.schedule_param}.resident_ctas == 0 ? "
            f"gridDim.x * gridDim.y * gridDim.z : {options.schedule_param}.resident_ctas;\n"
            f"  for (uint32_t __bs_task_id = __bs_cta; __bs_task_id < {options.schedule_param}.task_count; "
            "__bs_task_id += __bs_stride) {\n"
            f"    block_sche::RTask {options.task_param} = {options.schedule_param}.tasks[__bs_task_id];\n"
            f"{_trace_statement(options.trace, options.task_param, 'block_sche::current_sm_id()')}"
            f"    {device_name}({call_args});\n"
            "  }\n"
        )
    persistent_kernel = (
        f"{kernel.template_prefix}"
        f"__global__ void{launch_bounds_suffix} {persistent_name}({persistent_params}) "
        "{\n"
        f"{persistent_body}"
        "}\n"
    )
    parts = [device_function, persistent_kernel]
    if options.host_launcher:
        parts.append(
            _emit_host_launcher(
                kernel=kernel,
                persistent_name=persistent_name,
                params=params,
                param_names=param_names,
                trace=options.trace,
            )
        )
    return "\n".join(parts)


def _kernel_launch_bounds(kernel: Kernel) -> str:
    attr_start = kernel.qualifier_prefix.find("__launch_bounds__")
    if attr_start == -1:
        return ""
    call_start = _skip_ws(kernel.qualifier_prefix, attr_start + len("__launch_bounds__"))
    if call_start >= len(kernel.qualifier_prefix) or kernel.qualifier_prefix[call_start] != "(":
        return ""
    call_end = _find_matching(kernel.qualifier_prefix, call_start, "(", ")")
    return kernel.qualifier_prefix[attr_start : call_end + 1].strip()


def _validate_kernel_body(body: str, kernel_name: str) -> None:
    unsupported = {
        "cooperative_groups::this_grid": "cooperative grid synchronization is not preserved by rTask conversion",
        "cudaLaunchDevice": "dynamic parallelism launches are not rewritten",
        "<<<": "device-side CUDA launches are not rewritten",
    }
    for needle, reason in unsupported.items():
        if _contains_code_token(body, needle):
            raise TransformError(f"kernel {kernel_name!r} is unsupported: {reason}")


def _code_position_mask(source: str) -> List[bool]:
    mask = [False] * len(source)
    i = 0
    state = "code"
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""

        if state == "line_comment":
            if ch == "\n":
                mask[i] = True
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                i += 1
                state = "code"
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "code"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "code"
        else:
            if ch == "/" and nxt == "/":
                i += 1
                state = "line_comment"
            elif ch == "/" and nxt == "*":
                i += 1
                state = "block_comment"
            elif ch == '"':
                mask[i] = True
                state = "string"
            elif ch == "'":
                mask[i] = True
                state = "char"
            else:
                mask[i] = True
        i += 1
    return mask


def _contains_code_token(source: str, needle: str) -> bool:
    mask = _code_position_mask(source)
    start = 0
    while True:
        pos = source.find(needle, start)
        if pos == -1:
            return False
        if pos < len(mask) and mask[pos]:
            return True
        start = pos + 1


def _find_matching(source: str, start: int, open_ch: str, close_ch: str) -> int:
    if source[start] != open_ch:
        raise TransformError(f"expected {open_ch!r} at offset {start}")

    depth = 0
    i = start
    state = "code"
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""

        if state == "line_comment":
            if ch == "\n":
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "code"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "code"
        else:
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block_comment"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        i += 1

    raise TransformError(f"unmatched {open_ch!r} at offset {start}")


def _skip_ws(source: str, start: int) -> int:
    i = start
    while i < len(source) and source[i].isspace():
        i += 1
    return i


def _split_params(params: str) -> List[str]:
    params = params.strip()
    if not params or params == "void":
        return []

    out: List[str] = []
    start = 0
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    state = "code"

    for i, ch in enumerate(params):
        nxt = params[i + 1] if i + 1 < len(params) else ""
        if state == "line_comment":
            if ch == "\n":
                state = "code"
            continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
            continue
        if state == "string":
            if ch == "\\":
                continue
            if ch == '"':
                state = "code"
            continue
        if state == "char":
            if ch == "\\":
                continue
            if ch == "'":
                state = "code"
            continue

        if ch == "/" and nxt == "/":
            state = "line_comment"
        elif ch == "/" and nxt == "*":
            state = "block_comment"
        elif ch == '"':
            state = "string"
        elif ch == "'":
            state = "char"
        elif ch == "<":
            depth_angle += 1
        elif ch == ">" and depth_angle:
            depth_angle -= 1
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == "," and depth_angle == 0 and depth_paren == 0 and depth_bracket == 0:
            out.append(params[start:i].strip())
            start = i + 1

    out.append(params[start:].strip())
    return out


def _param_name(param: str) -> str:
    param = re.sub(r"=\s*.*$", "", param.strip())
    param = re.sub(r"\[[^\]]*\]\s*$", "", param)
    match = re.search(rf"({_IDENT})\s*$", param)
    if not match:
        raise TransformError(f"could not infer parameter name from {param!r}")
    name = match.group(1)
    if name in {"const", "volatile", "restrict", "__restrict__", "__restrict"}:
        raise TransformError(f"could not infer parameter name from {param!r}")
    return name


def _join_params(params: Sequence[str]) -> str:
    clean = [param.strip() for param in params if param.strip() and param.strip() != "void"]
    return ", ".join(clean) if clean else "void"


def _wrap_with_virtual_cuda_indices(body: str, task_param: str, schedule_param: str) -> str:
    return (
        f"  const uint3 __bs_blockIdx = {{{task_param}.block.x, {task_param}.block.y, "
        f"{task_param}.block.z}};\n"
        f"  const dim3 __bs_gridDim({schedule_param}.logical_grid.x, "
        f"{schedule_param}.logical_grid.y, {schedule_param}.logical_grid.z);\n"
        "#define blockIdx __bs_blockIdx\n"
        "#define gridDim __bs_gridDim\n"
        f"{body}\n"
        "#undef gridDim\n"
        "#undef blockIdx"
    )


def _trace_statement(enabled: bool, task_param: str, sm_expr: str) -> str:
    if not enabled:
        return ""
    return f"    block_sche::record_trace(__bs_trace, {task_param}, {sm_expr});\n"


def _emit_host_launcher(
    kernel: Kernel,
    persistent_name: str,
    params: Sequence[str],
    param_names: Sequence[str],
    trace: bool,
) -> str:
    launcher_name = f"{kernel.name}_launch"
    launcher_params = [
        "block_sche::PreparedSchedule& __bs_prepared",
    ]
    if trace:
        launcher_params.append("block_sche::PreparedTrace& __bs_trace")
    launcher_params.extend(
        [
            "dim3 __bs_block_dim",
            "cudaStream_t __bs_stream",
            *params,
        ]
    )
    persistent_args = ["__bs_prepared.device_view()"]
    if trace:
        persistent_args.append("__bs_trace.device_view()")
    persistent_args.extend(param_names)
    reset_trace = ""
    if trace:
        reset_trace = (
            "  __bs_err = __bs_trace.reset(__bs_stream);\n"
            "  if (__bs_err != cudaSuccess) return __bs_err;\n"
        )
    return (
        f"{kernel.template_prefix}"
        f"cudaError_t {launcher_name}({_join_params(launcher_params)}) "
        "{\n"
        "  cudaError_t __bs_err = __bs_prepared.reset_cursors(__bs_stream);\n"
        "  if (__bs_err != cudaSuccess) return __bs_err;\n"
        f"{reset_trace}"
        "  const auto __bs_launch = __bs_prepared.launch_config(__bs_block_dim);\n"
        f"  {persistent_name}<<<__bs_launch.persistent_grid, __bs_launch.block_dim,\n"
        f"                 __bs_launch.shared_memory_bytes, __bs_stream>>>"
        f"({', '.join(persistent_args)});\n"
        "  return cudaGetLastError();\n"
        "}\n"
    )
