#include "inference/npz.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace wm::infer {

namespace {

[[noreturn]] void fail(const std::string& what, const std::string& why) {
    throw std::runtime_error(what + ": " + why);
}

uint16_t u16(const char* p) {
    return static_cast<uint16_t>(static_cast<unsigned char>(p[0]) |
                                 static_cast<unsigned char>(p[1]) << 8);
}
uint32_t u32(const char* p) { return u16(p) | static_cast<uint32_t>(u16(p + 2)) << 16; }
uint64_t u64(const char* p) { return u32(p) | static_cast<uint64_t>(u32(p + 4)) << 32; }

size_t element_size(const std::string& dtype) {
    if (dtype == "<f4" || dtype == "<i4") return 4;
    if (dtype == "<i2") return 2;
    if (dtype == "|i1" || dtype == "|u1") return 1;
    return 0;
}

// The text after `key` in the .npy header dict, e.g. key "'descr':".
std::string header_field(const std::string& header, const std::string& key,
                         const std::string& what) {
    const size_t at = header.find(key);
    if (at == std::string::npos) fail(what, "npy header has no " + key);
    return header.substr(at + key.size());
}

}  // namespace

int64_t NpyArray::size() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

template <class T>
std::vector<T> NpyArray::as(const char* expected_dtype) const {
    if (dtype != expected_dtype)
        throw std::runtime_error("array has type " + dtype + ", expected " + expected_dtype);
    std::vector<T> out(static_cast<size_t>(size()));
    std::memcpy(out.data(), data.data(), out.size() * sizeof(T));
    return out;
}

template std::vector<float> NpyArray::as<float>(const char*) const;
template std::vector<int8_t> NpyArray::as<int8_t>(const char*) const;
template std::vector<int16_t> NpyArray::as<int16_t>(const char*) const;
template std::vector<int32_t> NpyArray::as<int32_t>(const char*) const;
template std::vector<uint8_t> NpyArray::as<uint8_t>(const char*) const;

// .npy layout: "\x93NUMPY", major version, minor version, header length
// (2 bytes in version 1, 4 bytes in versions 2 and 3), then a Python dict as
// text, e.g. {'descr': '<f4', 'fortran_order': False, 'shape': (256, 769), }
// and finally the raw array bytes.
NpyArray parse_npy(const char* bytes, size_t size, const std::string& what) {
    if (size < 10 || std::memcmp(bytes, "\x93NUMPY", 6) != 0) fail(what, "not a .npy file");
    const int major = static_cast<unsigned char>(bytes[6]);
    size_t header_len, start;
    if (major == 1) {
        header_len = u16(bytes + 8);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (size < 12) fail(what, "truncated .npy header");
        header_len = u32(bytes + 8);
        start = 12;
    } else {
        fail(what, "unsupported .npy version " + std::to_string(major));
    }
    if (start + header_len > size) fail(what, "truncated .npy header");
    const std::string header(bytes + start, header_len);

    NpyArray arr;
    // 'descr': '<f4'
    std::string descr = header_field(header, "'descr':", what);
    const size_t q1 = descr.find('\''), q2 = descr.find('\'', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) fail(what, "bad descr");
    arr.dtype = descr.substr(q1 + 1, q2 - q1 - 1);
    if (element_size(arr.dtype) == 0) fail(what, "unsupported type " + arr.dtype);

    // 'fortran_order': False (column order would need transposing; never used)
    if (header_field(header, "'fortran_order':", what).find("False") > 2)
        fail(what, "Fortran-ordered arrays aren't supported");

    // 'shape': (256, 769)  or (3,)  or ()
    std::string shape = header_field(header, "'shape':", what);
    const size_t open = shape.find('('), close = shape.find(')');
    if (open == std::string::npos || close == std::string::npos) fail(what, "bad shape");
    std::string dims = shape.substr(open + 1, close - open - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) ++pos;
        if (pos >= dims.size()) break;
        size_t used = 0;
        arr.shape.push_back(std::stoll(dims.substr(pos), &used));
        pos += used;
    }

    const size_t nbytes = static_cast<size_t>(arr.size()) * element_size(arr.dtype);
    const size_t data_start = start + header_len;
    if (data_start + nbytes > size) fail(what, "array data is truncated");
    arr.data.assign(bytes + data_start, bytes + data_start + nbytes);
    return arr;
}

// A .npz file is a zip archive. Its table of contents (the "central
// directory") sits at the end of the file and gives each member's name and
// where its data starts. NumPy writes zip64 records (for files over 4 GB),
// so the 64-bit sizes and offsets are read from the zip64 "extra" field when
// the 32-bit ones are set to 0xFFFFFFFF.
std::map<std::string, NpyArray> load_npz(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(path, "can't open file");
    std::vector<char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t n = file.size();
    const char* b = file.data();

    // End-of-central-directory record: signature PK\5\6, at most 64 KB from
    // the end (it may be followed by a comment).
    size_t eocd = std::string::npos;
    for (size_t i = n >= 22 ? n - 22 : 0; i + 22 <= n; --i) {
        if (u32(b + i) == 0x06054b50) {
            eocd = i;
            break;
        }
        if (i == 0 || n - i > 65557) break;
    }
    if (eocd == std::string::npos) fail(path, "not a zip (.npz) file");
    uint64_t entries = u16(b + eocd + 10);
    uint64_t dir_offset = u32(b + eocd + 16);

    // Zip64: the real values are in the zip64 end record, found via its locator.
    if ((dir_offset == 0xFFFFFFFF || entries == 0xFFFF) && eocd >= 20 &&
        u32(b + eocd - 20) == 0x07064b50) {
        const uint64_t z64 = u64(b + eocd - 20 + 8);
        if (z64 + 56 > n || u32(b + z64) != 0x06064b50) fail(path, "bad zip64 record");
        entries = u64(b + z64 + 32);
        dir_offset = u64(b + z64 + 48);
    }

    std::map<std::string, NpyArray> arrays;
    size_t p = static_cast<size_t>(dir_offset);
    for (uint64_t e = 0; e < entries; ++e) {
        if (p + 46 > n || u32(b + p) != 0x02014b50) fail(path, "bad zip directory");
        const uint16_t method = u16(b + p + 10);
        uint64_t csize = u32(b + p + 20);
        uint64_t usize = u32(b + p + 24);
        const uint16_t name_len = u16(b + p + 28), extra_len = u16(b + p + 30),
                       comment_len = u16(b + p + 32);
        uint64_t local = u32(b + p + 42);
        std::string name(b + p + 46, name_len);

        // Zip64 extra field (id 1): 64-bit values, only for fields set to 0xFFFFFFFF,
        // in the order: uncompressed size, compressed size, local header offset.
        const char* extra = b + p + 46 + name_len;
        for (size_t x = 0; x + 4 <= extra_len;) {
            const uint16_t id = u16(extra + x), len = u16(extra + x + 2);
            if (id == 1) {
                const char* v = extra + x + 4;
                if (usize == 0xFFFFFFFF) { usize = u64(v); v += 8; }
                if (csize == 0xFFFFFFFF) { csize = u64(v); v += 8; }
                if (local == 0xFFFFFFFF) { local = u64(v); }
            }
            x += 4 + len;
        }
        p += 46 + name_len + extra_len + comment_len;

        if (method != 0)
            fail(path, name + " is compressed; save with np.savez, not np.savez_compressed");
        if (local + 30 > n || u32(b + local) != 0x04034b50) fail(path, "bad zip entry " + name);
        const size_t data = static_cast<size_t>(local) + 30 + u16(b + local + 26) +
                            u16(b + local + 28);
        if (data + usize > n) fail(path, name + " is truncated");

        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0)
            name.resize(name.size() - 4);
        arrays[name] = parse_npy(b + data, static_cast<size_t>(usize), path + "/" + name);
    }
    return arrays;
}

}  // namespace wm::infer
