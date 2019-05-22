package crc32util

import (
	"encoding/binary"
	"hash/crc32"
	"io"

	"github.com/qiniu/errors"
	qio "github.com/qiniu/io"
)

const (
	chunkBits = 16 // 64K
	chunkLen  = (1 << chunkBits) - 4
)

const (
	BufSize = chunkLen + 4
)

var (
	ErrUnmatchedChecksum = errors.New("unmatched checksum")
	ErrClosed            = errors.New("has already been closed")
)

// -----------------------------------------

func EncodeSize(fsize int64) int64 {

	chunkCount := (fsize + (chunkLen - 1)) / chunkLen
	return fsize + 4*chunkCount
}

func DecodeSize(totalSize int64) int64 {

	chunkCount := (totalSize + (BufSize - 1)) / BufSize
	return totalSize - 4*chunkCount
}

// -----------------------------------------

type ReaderError struct {
	error
}

type WriterError struct {
	error
}

func Encode(w io.Writer, in io.Reader, fsize int64, chunk []byte) (err error) {

	if chunk == nil {
		chunk = make([]byte, BufSize)
	} else if len(chunk) != BufSize {
		panic("crc32util.Encode failed: invalid len(chunk)")
	}

	i := 0
	for fsize >= chunkLen {
		_, err = io.ReadFull(in, chunk[4:])
		if err != nil {
			return ReaderError{err}
		}
		crc := crc32.ChecksumIEEE(chunk[4:])
		binary.LittleEndian.PutUint32(chunk, crc)
		_, err = w.Write(chunk)
		if err != nil {
			return WriterError{err}
		}
		fsize -= chunkLen
		i++
	}

	if fsize > 0 {
		n := fsize + 4
		_, err = io.ReadFull(in, chunk[4:n])
		if err != nil {
			return ReaderError{err}
		}
		crc := crc32.ChecksumIEEE(chunk[4:n])
		binary.LittleEndian.PutUint32(chunk, crc)
		_, err = w.Write(chunk[:n])
		if err != nil {
			err = WriterError{err}
		}
	}
	return
}

// ---------------------------------------------------------------------------

type ReaderWriterAt interface {
	io.ReaderAt
	io.WriterAt
}

//
// 对于数据开始在(rw io.ReaderWriterAt, base int64) 的文件大小为 fsize 的做 crc32 冗余校验的文件
// 我们要往它后面追加 size 大小的数据
func AppendEncode(rw ReaderWriterAt, base int64, fsize int64, in io.Reader, size int64, chunk []byte) (err error) {

	if chunk == nil {
		chunk = make([]byte, BufSize)
	} else if len(chunk) != BufSize {
		panic("crc32util.Encode failed: invalid len(chunk)")
	}

	offset := base + EncodeSize(fsize)
	if oldSize := fsize % chunkLen; oldSize > 0 {
		// 旧文件的最后一个 chunk 需要特殊处理。
		// 处理流程为：读取旧内容、写入新内容、写入总 crc32。
		r := RangeDecoder(rw, base, chunk, fsize-oldSize, fsize, fsize)
		_, err = io.ReadFull(r, chunk[4:4+oldSize])
		if err != nil {
			// 从 rw 读失败，认为是 writer 错误。
			return WriterError{err}
		}
		addSize := chunkLen - oldSize
		if addSize > size {
			addSize = size
		}
		add := chunk[4+oldSize : 4+oldSize+addSize]
		_, err = io.ReadFull(in, add)
		if err != nil {
			return ReaderError{err}
		}
		// 如果 header 写成功但 data 写失败，这个 chunk 就无法正常读写了。
		// 因此下面的操作是先写 data 再写 header。
		_, err = rw.WriteAt(add, offset)
		if err != nil {
			return WriterError{err}
		}
		crc := crc32.ChecksumIEEE(chunk[4 : 4+oldSize+addSize])
		pos := base + (fsize/chunkLen)<<chunkBits
		defer func() {
			if err != nil {
				return
			}
			binary.LittleEndian.PutUint32(chunk[:4], crc)
			_, err = rw.WriteAt(chunk[:4], pos)
			if err != nil {
				err = WriterError{err}
			}
		}()
		size -= addSize
		offset += addSize
	}
	if size == 0 {
		return nil
	}
	w := &qio.Writer{
		WriterAt: rw,
		Offset:   offset,
	}
	return Encode(w, in, size, chunk)
}

// ---------------------------------------------------------------------------
type simpleEncoder struct {
	chunk []byte
	in    io.Reader
	off   int
}

// 支持in.size = 0, 不支持 in = nil的情况
func SimpleEncoder(in io.Reader, chunk []byte) (enc *simpleEncoder) {
	if chunk == nil {
		chunk = make([]byte, BufSize)
	} else if len(chunk) != BufSize {
		panic("crc32util.Decoder failed: invalid len(chunk)")
	}

	enc = &simpleEncoder{chunk, in, BufSize}
	return
}

// 读到错误的块，即使前面有数据，也会抛弃
func (r *simpleEncoder) Read(b []byte) (n int, err error) {

	if r.off == len(r.chunk) {
		err = r.fetch()
		if err != nil {
			return
		}
	}

	n = copy(b, r.chunk[r.off:])
	r.off += n
	return
}

// assert：r.read 出现err之后，再调用r.read不能读出数据。
func (r *simpleEncoder) fetch() (err error) {

	var n int
	n, err = ReadSize(r.in, r.chunk[4:])
	if err != nil {
		return
	}

	crc := crc32.ChecksumIEEE(r.chunk[4 : n+4])
	binary.LittleEndian.PutUint32(r.chunk, crc)
	r.off = 0
	r.chunk = r.chunk[:n+4]
	return
}

// 一次读取len(buf)个数据
// 如果读取n(n > 0, n可以小于 len(buf))个字节后遇到了EOF，那么返回这n个字节，并且err == nil
// 下一次再返回 err == EOF && n == 0
// 如果读取过程中遇到了其他错误，那么直接返回错误。
// 不会同时返回 err = nil && n = 0
func ReadSize(r io.Reader, buf []byte) (n int, err error) {

	size := len(buf)
	for n < size && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	if err == io.EOF && n != 0 {
		err = nil
	}
	return
}

type simpleDecoder struct {
	chunk []byte
	in    io.Reader
	off   int
}

// 不支持in = nil的情况
// in size 的正确取值范围是 0 ||(4 + 64K * n, 64K + 64K * n]
func SimpleDecoder(in io.Reader, chunk []byte) (dec *simpleDecoder) {
	if chunk == nil {
		chunk = make([]byte, BufSize)
	} else if len(chunk) != BufSize {
		panic("crc32util.Decoder failed: invalid len(chunk)")
	}

	dec = &simpleDecoder{chunk, in, BufSize}
	return
}

func (r *simpleDecoder) Read(b []byte) (n int, err error) {

	if r.off == len(r.chunk) {
		err = r.fetch()
		if err != nil {
			return
		}
	}

	n = copy(b, r.chunk[r.off:])
	r.off += n
	return
}

func (r *simpleDecoder) fetch() (err error) {

	var n int
	n, err = ReadSize(r.in, r.chunk)
	if err != nil {
		return
	}

	if n <= 4 {
		err = errors.Info(ErrUnmatchedChecksum, "crc32util.decode")
		return
	}
	crc := crc32.ChecksumIEEE(r.chunk[4:n])
	if binary.LittleEndian.Uint32(r.chunk) != crc {
		err = errors.Info(ErrUnmatchedChecksum, "crc32util.decode")
		return
	}
	r.chunk = r.chunk[:n]
	r.off = 4
	return
}

// 对于写入的数据加入 crc32校验
// 调用 close 表示结束。
type encodeWriter struct {
	w     io.Writer
	chunk []byte // 用于保存每次 write 之后剩余的数据
	off   int    // 用于保存每次 write 之后 chunk 的index
}

func NewEncodeWriteCloser(w io.Writer) (ewc *encodeWriter) {
	chunk := make([]byte, BufSize)
	ewc = &encodeWriter{
		w:     w,
		chunk: chunk,
		off:   4,
	}
	return
}

func (w *encodeWriter) Write(p []byte) (n int, err error) {
	offset := w.off
	size := (offset - 4) + len(p)
	var pfrom int = 0
	var pto int
	for size >= chunkLen {
		pto = BufSize - offset + pfrom
		copy(w.chunk[offset:], p[pfrom:pto])
		crc := crc32.ChecksumIEEE(w.chunk[4:])
		binary.LittleEndian.PutUint32(w.chunk, crc)
		_, err = w.w.Write(w.chunk)
		if err != nil {
			return 0, err
		}

		// 处理已经写完的数据
		offset = 4
		pfrom = pto
		size -= chunkLen
	}

	n1 := copy(w.chunk[offset:], p[pfrom:])
	w.off = offset + n1
	return len(p), nil
}

func (w *encodeWriter) CloseWithError(err error) error {
	if err == nil {
		return w.Close()
	}
	return nil
}

// 需要调用 close 方法，写出最后的数据。
// 失败的时候，调用 closewitherror，而不是 close，否则客户端可能会得到 unmatch checksum
func (w *encodeWriter) Close() (err error) {

	if w.off > 4 {
		// 将缓冲中的数据写出
		crc := crc32.ChecksumIEEE(w.chunk[4:w.off])
		binary.LittleEndian.PutUint32(w.chunk, crc)
		_, err = w.w.Write(w.chunk[:w.off])
		if err != nil {
			return
		}
		w.off = 4
	}
	return nil
}

type decoder struct { // raw+crc32 input => raw input
	chunk   []byte
	in      io.Reader
	lastErr error
	off     int
	left    int64
}

// n 代表 fsize 不是encode size
// 支持n=0的情况
func Decoder(in io.Reader, n int64, chunk []byte) (dec *decoder) {

	if chunk == nil {
		chunk = make([]byte, BufSize)
	} else if len(chunk) != BufSize {
		panic("crc32util.Decoder failed: invalid len(chunk)")
	}

	dec = &decoder{chunk, in, nil, BufSize, n}
	return
}

func (r *decoder) fetch() {

	min := len(r.chunk)
	if r.left+4 < int64(min) {
		min = int(r.left + 4)
	}
	var n2 int
	n2, r.lastErr = io.ReadAtLeast(r.in, r.chunk, min)
	if r.lastErr != nil {
		if r.lastErr == io.EOF {
			r.lastErr = io.ErrUnexpectedEOF
		}
		return
	}
	crc := crc32.ChecksumIEEE(r.chunk[4:n2])
	if binary.LittleEndian.Uint32(r.chunk) != crc {
		r.lastErr = errors.Info(ErrUnmatchedChecksum, "crc32util.decode")
		return
	}
	r.chunk = r.chunk[:n2]
	r.off = 4
}

func (r *decoder) Read(b []byte) (n int, err error) {

	if r.off == len(r.chunk) {
		if r.lastErr != nil {
			err = r.lastErr
			return
		}
		if r.left == 0 {
			err = io.EOF
			return
		}
		r.fetch()
	}
	n = copy(b, r.chunk[r.off:])
	r.off += n
	r.left -= int64(n)
	return
}

// ---------------------------------------------------------------------------

//
// 对于数据开始在 (in io.ReaderAt, base int64)，文件大小为 fsize 的做 crc32 冗余校验的文件，我们
// 要读取其中 [from, to) 范围的数据。
//
func RangeDecoder(in io.ReaderAt, base int64, chunk []byte, from, to, fsize int64) io.Reader {

	fromBase := (from / chunkLen) << chunkBits
	encodedSize := EncodeSize(fsize) - fromBase
	sect := io.NewSectionReader(in, base+fromBase, encodedSize)
	dec := Decoder(sect, DecodeSize(encodedSize), chunk)
	if (from == 0 || from%chunkLen == 0) && to >= fsize {
		return dec
	}
	return newSectionReader(dec, from%chunkLen, to-from)
}

// ---------------------------------------------------------------------------

func decodeAt(w io.Writer, in io.ReaderAt, chunk []byte, idx int64, ifrom, ito int) (err error) {

	n, err := in.ReadAt(chunk, idx<<chunkBits)
	if err != nil {
		if err != io.EOF {
			return
		}
	}
	if n <= 4 {
		if n == 0 {
			return io.EOF
		}
		err = errors.Info(io.ErrUnexpectedEOF, "crc32util.Decode", "n:", n)
		return
	}

	crc := crc32.ChecksumIEEE(chunk[4:n])
	if binary.LittleEndian.Uint32(chunk) != crc {
		err = errors.Info(ErrUnmatchedChecksum, "crc32util.Decode")
		return
	}

	ifrom += 4
	ito += 4
	if ito > n {
		ito = n
	}
	if ifrom >= ito {
		return io.EOF
	}
	_, err = w.Write(chunk[ifrom:ito])
	return
}

func DecodeRange(w io.Writer, in io.ReaderAt, chunk []byte, from, to int64) (err error) {

	if from >= to {
		return
	}

	if chunk == nil {
		chunk = make([]byte, BufSize)
	} else if len(chunk) != BufSize {
		panic("crc32util.Decode failed: invalid len(chunk)")
	}

	fromIdx, toIdx := from/chunkLen, to/chunkLen
	fromOff, toOff := int(from%chunkLen), int(to%chunkLen)
	if fromIdx == toIdx { // 只有一行
		return decodeAt(w, in, chunk, fromIdx, fromOff, toOff)
	}
	for fromIdx < toIdx {
		err = decodeAt(w, in, chunk, fromIdx, fromOff, chunkLen)
		if err != nil {
			return
		}
		fromIdx++
		fromOff = 0
	}
	if toOff > 0 {
		err = decodeAt(w, in, chunk, fromIdx, 0, toOff)
	}
	return
}

// ---------------------------------------------------------------------------
