from pathlib import Path

import torch

from scripts.llama_cpp.convert_direction_to_gguf import write_direction_gguf


def _read_u32(buf: bytes, off: int):
    import struct
    return struct.unpack_from('<I', buf, off)[0], off + 4


def _read_u64(buf: bytes, off: int):
    import struct
    return struct.unpack_from('<Q', buf, off)[0], off + 8


def _read_string(buf: bytes, off: int):
    n, off = _read_u64(buf, off)
    s = buf[off:off+n].decode('utf-8')
    return s, off + n


def test_write_direction_gguf_emits_direction_tensors(tmp_path: Path):
    out = tmp_path / 'qwen25-7b.gguf'
    vec = torch.arange(8, dtype=torch.float32)
    write_direction_gguf(out, vec, [10, 11, 12], dir_layer=11, source='test-key')

    buf = out.read_bytes()
    off = 0
    magic, off = _read_u32(buf, off)
    version, off = _read_u32(buf, off)
    assert magic == 0x46554747
    assert version == 3

    n_tensors, off = _read_u64(buf, off)
    n_kv, off = _read_u64(buf, off)
    assert n_tensors == 3
    assert n_kv >= 4

    # Skip KV section generically
    for _ in range(n_kv):
        _key, off = _read_string(buf, off)
        typ, off = _read_u32(buf, off)
        if typ == 8:  # string
            _val, off = _read_string(buf, off)
        elif typ == 4:  # uint32
            _val, off = _read_u32(buf, off)
        elif typ == 10:  # uint64
            _val, off = _read_u64(buf, off)
        elif typ == 9:  # array
            elem_type, off = _read_u32(buf, off)
            length, off = _read_u64(buf, off)
            assert elem_type == 4
            for _ in range(length):
                _v, off = _read_u32(buf, off)
        else:
            raise AssertionError(f'unexpected gguf kv type {typ}')

    names = []
    for _ in range(n_tensors):
        name, off = _read_string(buf, off)
        names.append(name)
        n_dims, off = _read_u32(buf, off)
        assert n_dims == 1
        dim0, off = _read_u64(buf, off)
        assert dim0 == 8
        ggml_type, off = _read_u32(buf, off)
        tensor_off, off = _read_u64(buf, off)
        assert ggml_type == 0
        assert tensor_off % 4 == 0

    assert names == ['direction.10', 'direction.11', 'direction.12']


def _read_float32(buf: bytes, off: int):
    import struct
    return struct.unpack_from('<f', buf, off)[0], off + 4


def _parse_kv(buf, off, n_kv):
    """Parse GGUF key-value pairs, return as dict."""
    kvs = {}
    for _ in range(n_kv):
        key, off = _read_string(buf, off)
        typ, off = _read_u32(buf, off)
        if typ == 8:  # string
            val, off = _read_string(buf, off)
        elif typ == 4:  # uint32
            val, off = _read_u32(buf, off)
        elif typ == 6:  # float32
            val, off = _read_float32(buf, off)
        elif typ == 10:  # uint64
            val, off = _read_u64(buf, off)
        elif typ == 9:  # array
            elem_type, off = _read_u32(buf, off)
            length, off = _read_u64(buf, off)
            val = []
            for _ in range(length):
                if elem_type == 4:
                    v, off = _read_u32(buf, off)
                    val.append(v)
                else:
                    raise AssertionError(f'unexpected array elem type {elem_type}')
        else:
            raise AssertionError(f'unexpected gguf kv type {typ}')
        kvs[key] = val
    return kvs, off


def test_steer_mode_emits_steer_adapter_type_and_alpha(tmp_path: Path):
    out = tmp_path / 'llama8b_steer.gguf'
    vec = torch.arange(8, dtype=torch.float32)
    write_direction_gguf(out, vec, [20, 21, 22], dir_layer=21,
                         source='test-steer', mode='steer', alpha=2.5)

    buf = out.read_bytes()
    off = 0
    _magic, off = _read_u32(buf, off)
    _version, off = _read_u32(buf, off)
    n_tensors, off = _read_u64(buf, off)
    n_kv, off = _read_u64(buf, off)

    kvs, _off = _parse_kv(buf, off, n_kv)

    assert kvs["ungag.adapter_type"] == "steer_direction"
    assert abs(kvs["ungag.alpha"] - 2.5) < 1e-6
    assert kvs["ungag.slab"] == [20, 21, 22]
    assert n_tensors == 3


def test_project_mode_emits_proj_out_adapter_type(tmp_path: Path):
    out = tmp_path / 'yi34b_proj.gguf'
    vec = torch.arange(8, dtype=torch.float32)
    write_direction_gguf(out, vec, [29, 30, 31], dir_layer=30,
                         source='test-proj', mode='project')

    buf = out.read_bytes()
    off = 0
    _magic, off = _read_u32(buf, off)
    _version, off = _read_u32(buf, off)
    _n_tensors, off = _read_u64(buf, off)
    n_kv, off = _read_u64(buf, off)

    kvs, _off = _parse_kv(buf, off, n_kv)

    assert kvs["ungag.adapter_type"] == "proj_out_direction"
    assert "ungag.alpha" not in kvs
