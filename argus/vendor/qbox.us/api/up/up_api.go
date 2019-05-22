package up

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"hash/crc32"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/qiniu/api/auth/digest"
	. "github.com/qiniu/api/conf"
	"github.com/qiniu/log.v1"
	"qbox.us/api"
	. "qbox.us/api/conf"
	"qbox.us/errors"
	"qbox.us/rpc"
)

// ----------------------------------------------------------

const (
	InvalidCtx           = 701 // UP: 无效的上下文(bput)，可能情况：Ctx非法或者已经被淘汰（太久未使用）
	StatusCallbackFailed = 579 // UP: 上传成功，回调失败
)

type FileType uint32

const (
	TypeNormal = iota
	TypeLine
)

// ----------------------------------------------------------

type AuthPolicy struct {
	Scope               string   `json:"scope"`
	CallbackUrl         string   `json:"callbackUrl,omitempty"`
	CallbackHost        string   `json:"callbackHost,omitempty"`
	CallbackBodyType    string   `json:"callbackBodyType,omitempty"`
	CallbackBody        string   `json:"callbackBody,omitempty"`
	CallbackFetchKey    uint16   `json:"callbackFetchKey,omitempty"` // 先回调取得key再改名 https://pm.qbox.me/issues/11851
	CallbackTimeout     uint16   `json:"callbackTimeout,omitempty"`  // 允许自定义超时需求 https://pm.qbox.me/issues/21576
	Customer            string   `json:"customer,omitempty"`
	EndUser             string   `json:"endUser,omitempty"`
	Transform           string   `json:"transform,omitempty"`
	FopTimeout          uint32   `json:"fopTimeout,omitempty"`
	Deadline            int64    `json:"deadline"`         // 截止时间（以秒为单位）原来是uint32 上限为到2106年 如果用户设置过期时间超过了这个上限就会鉴权失败 请各单位如果编译不过自行调整https://pm.qbox.me/issues/25718
	Escape              uint16   `json:"escape,omitempty"` // 是否允许存在转义符号
	DetectMime          uint16   `json:"detectMime,omitempty"`
	Exclusive           uint16   `json:"exclusive,omitempty"`  // 若为非0, 即使Scope为"Bucket:key"的形式也是insert only
	InsertOnly          uint16   `json:"insertOnly,omitempty"` // Exclusive 的别名
	ReturnBody          string   `json:"returnBody,omitempty"`
	SignReturnBody      uint16   `json:"signReturnBody,omitempty"` // 默认不开启签名，需要用户的 AK SK
	ReturnURL           string   `json:"returnUrl,omitempty"`
	FsizeMin            int64    `json:"fsizeMin,omitempty"`
	FsizeLimit          int64    `json:"fsizeLimit,omitempty"`
	MimeLimit           string   `json:"mimeLimit,omitempty"`
	SaveKey             string   `json:"saveKey,omitempty"`
	PersistentOps       string   `json:"persistentOps,omitempty"`
	PersistentNotifyUrl string   `json:"persistentNotifyUrl,omitempty"`
	PersistentPipeline  string   `json:"persistentPipeline,omitempty"`
	Checksum            string   `json:"checksum,omitempty"`
	Accesses            []string `json:"accesses,omitempty"`
	DeleteAfterDays     uint32   `json:"deleteAfterDays,omitempty"`
	FileType            FileType `json:"file_type,omitempty"`
	NotifyQueue         string   `json:"notifyQueue,omitempty"`
	NotifyMessage       string   `json:"notifyMessage,omitempty"`
	NotifyMessageType   string   `json:"notifyMessageType,omitempty"`
}

func MakeAuthToken(key, secret []byte, auth *AuthPolicy) []byte {
	b, _ := json.Marshal(auth)
	mac := &digest.Mac{string(key), secret}
	return []byte(mac.SignWithData(b))
}

func MakeAuthTokenString(key, secret string, auth *AuthPolicy) string {
	b, _ := json.Marshal(auth)
	mac := &digest.Mac{key, []byte(secret)}
	return mac.SignWithData(b)
}

// ----------------------------------------------------------

// Transport implements http.RoundTripper. When configured with a valid
// Config and Token it can be used to make authenticated HTTP requests.
//
//	c := NewClient(token, nil)
//	...
//
type Transport struct {
	token string

	// Transport is the HTTP transport to use when making requests.
	// It will default to http.DefaultTransport if nil.
	// (It should never be an oauth.Transport.)
	transport http.RoundTripper
}

// RoundTrip executes a single HTTP transaction using the Transport's
// Token as authorization headers.
func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	req.Header.Set("Authorization", t.token)
	return t.transport.RoundTrip(req)
}

func (t *Transport) NestedObject() interface{} {
	return t.transport
}

func NewTransport(token string, transport http.RoundTripper) *Transport {
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &Transport{"UpToken " + token, transport}
}

func NewClient(token string, transport http.RoundTripper) *http.Client {
	t := NewTransport(token, transport)
	return &http.Client{Transport: t}
}

// ----------------------------------------------------------

type PutRet struct {
	Ctx       string `json:"ctx"`
	Checksum  string `json:"checksum"`
	Crc32     uint32 `json:"crc32"`
	Offset    uint32 `json:"offset"`
	Host      string `json:"host"` // 过时，兼容保留
	ExpiredAt int64  `json:"expired_at,omitempty"`
}

func Mkblock(c rpc.Client, blockSize int, body io.Reader, bodyLength int) (ret PutRet, code int, err error) {

	code, err = c.CallWithBinaryEx(
		&ret, UP_HOST+"/mkblk/"+strconv.Itoa(blockSize), "application/octet-stream", body, bodyLength)
	return
}

func GlbMkblock(upHost string, c rpc.Client, blockSize int, body io.Reader, bodyLength int) (ret PutRet, code int, err error) {

	code, err = c.CallWithBinaryEx(
		&ret, upHost+"/glb/mkblk/"+strconv.Itoa(blockSize), "application/octet-stream", body, bodyLength)
	return
}

func Blockput(c rpc.Client, ctx string, offset int, body io.Reader, bodyLength int) (ret PutRet, code int, err error) {

	code, err = c.CallWithBinaryEx(
		&ret, UP_HOST+"/bput/"+ctx+"/"+strconv.Itoa(offset), "application/octet-stream", body, bodyLength)
	return
}

func GlbBlockput(upHost string, c rpc.Client, ctx string, offset int, body io.Reader, bodyLength int) (ret PutRet, code int, err error) {

	code, err = c.CallWithBinaryEx(
		&ret, upHost+"/glb/bput/"+ctx+"/"+strconv.Itoa(offset), "application/octet-stream", body, bodyLength)
	return
}

// ----------------------------------------------------------

type BlockputProgress struct {
	Ctx      string
	Offset   int
	RestSize int
	Err      error
	Host     string
}

func ResumableBlockput(
	c rpc.Client, f io.ReaderAt, blockIdx int, blkSize, chunkSize, retryTimes int,
	prog *BlockputProgress, notify func(blockIdx int, prog *BlockputProgress)) (ret PutRet, code int, err error) {

	offbase := int64(blockIdx) << BLOCK_BITS
	h := crc32.NewIEEE()

	var bodyLength int

	if prog.Ctx == "" {

		if chunkSize < blkSize {
			bodyLength = chunkSize
		} else {
			bodyLength = blkSize
		}

		body1 := io.NewSectionReader(f, offbase, int64(bodyLength))
		body := io.TeeReader(body1, h)

		ret, code, err = Mkblock(c, blkSize, body, bodyLength)
		if err != nil {
			err = errors.Info(err, "ResumableBlockput: Mkblock failed").Detail(err)
			return
		}

		if ret.Crc32 != h.Sum32() {
			err = errors.Info(errors.ErrUnmatchedChecksum, "ResumableBlockput: invalid checksum").Detail(err)
			return
		}

		prog.Ctx = ret.Ctx
		prog.Offset = bodyLength
		prog.RestSize = blkSize - bodyLength

		notify(blockIdx, prog)

	} else if prog.Offset+prog.RestSize != blkSize {

		code, err = 400, errors.Info(api.EInvalidArgs, "ResumableBlockput")
		return
	}

	for prog.RestSize > 0 {

		if chunkSize < prog.RestSize {
			bodyLength = chunkSize
		} else {
			bodyLength = prog.RestSize
		}

		retry := retryTimes

	lzRetry:
		log.Debug("ResumableBlockput:", offbase, prog.Offset, bodyLength)
		body1 := io.NewSectionReader(f, offbase+int64(prog.Offset), int64(bodyLength))
		h.Reset()
		body := io.TeeReader(body1, h)

		ret, code, err = Blockput(c, prog.Ctx, prog.Offset, body, bodyLength)
		if err == nil {
			if ret.Crc32 == h.Sum32() {
				prog.Ctx = ret.Ctx
				prog.Offset += bodyLength
				prog.RestSize -= bodyLength
				notify(blockIdx, prog)
				continue
			} else {
				err = errors.Info(
					errors.ErrUnmatchedChecksum, "ResumableBlockput", "invalid checksum",
					offbase, prog.Offset, bodyLength).Detail(err).Warn()
			}
		} else {
			err = errors.Info(
				err, "ResumableBlockput", "Blockput failed",
				offbase, prog.Offset, bodyLength).Detail(err).Warn()
			if code == InvalidCtx {
				log.Warn("ResumableBlockput: invalid ctx, please retry")
				prog.Ctx = ""
				notify(blockIdx, prog)
				break
			}
		}
		if retry > 0 {
			retry--
			log.Info("ResumableBlockput retrying ...", err)
			goto lzRetry
		}
		break
	}
	return
}

type hostLock struct {
	sync.RWMutex
	needLock bool
	host     string
	err      error
}

func GlbResumableBlockput(
	hlock *hostLock, c rpc.Client, f io.ReaderAt, blockIdx int, blkSize, chunkSize, retryTimes int,
	prog *BlockputProgress, notify func(blockIdx int, prog *BlockputProgress)) (ret PutRet, code int, err error) {
	offbase := int64(blockIdx) << BLOCK_BITS
	h := crc32.NewIEEE()

	var bodyLength int
	if prog.Ctx == "" {

		if chunkSize < blkSize {
			bodyLength = chunkSize
		} else {
			bodyLength = blkSize
		}

		body1 := io.NewSectionReader(f, offbase, int64(bodyLength))
		body := io.TeeReader(body1, h)
		if blockIdx != 0 {
			hlock.RLock()
			defer hlock.RUnlock()
			if hlock.err != nil {
				err = hlock.err
				return
			}
		}

		ret, code, err = GlbMkblock(hlock.host, c, blkSize, body, bodyLength)
		if err != nil {
			err = errors.Info(err, "GlbResumableBlockput: Mkblock failed").Detail(err)
			return
		}
		if blockIdx == 0 {
			if hlock.needLock {
				hlock.host = ret.Host
				hlock.needLock = false
				hlock.Unlock()
			}
		}

		if ret.Crc32 != h.Sum32() {
			err = errors.Info(errors.ErrUnmatchedChecksum, "GlbResumableBlockput: invalid checksum").Detail(err)
			return
		}
		if ret.Host != hlock.host {
			err = errors.New("hosts are not the same")
			return
		}

		prog.Ctx = ret.Ctx
		prog.Offset = bodyLength
		prog.RestSize = blkSize - bodyLength
		prog.Host = hlock.host

		notify(blockIdx, prog)

	} else if prog.Offset+prog.RestSize != blkSize {

		code, err = 400, errors.Info(api.EInvalidArgs, "GlbResumableBlockput")
		return
	}

	for prog.RestSize > 0 {

		if chunkSize < prog.RestSize {
			bodyLength = chunkSize
		} else {
			bodyLength = prog.RestSize
		}

		retry := retryTimes
	lzRetry:
		log.Debug("Glb ResumableBlockput:", offbase, prog.Offset, bodyLength)
		body1 := io.NewSectionReader(f, offbase+int64(prog.Offset), int64(bodyLength))
		h.Reset()
		body := io.TeeReader(body1, h)

		ret, code, err = GlbBlockput(hlock.host, c, prog.Ctx, prog.Offset, body, bodyLength)
		if err == nil {
			if ret.Crc32 == h.Sum32() {
				prog.Ctx = ret.Ctx
				prog.Offset += bodyLength
				prog.RestSize -= bodyLength
				notify(blockIdx, prog)
				continue
			} else {
				err = errors.Info(
					errors.ErrUnmatchedChecksum, "ResumableBlockput", "invalid checksum",
					offbase, prog.Offset, bodyLength).Detail(err).Warn()
			}
		} else {
			err = errors.Info(
				err, "Glb ResumableBlockput", "Blockput failed",
				offbase, prog.Offset, bodyLength).Detail(err).Warn()
			if code == InvalidCtx {
				log.Warn("Glb ResumableBlockput: invalid ctx, please retry")
				prog.Ctx = ""
				notify(blockIdx, prog)
				break
			}
		}
		if retry > 0 {
			retry--
			log.Info("Glb ResumableBlockput retrying ...")
			goto lzRetry
		}
		break
	}
	return
}

// ----------------------------------------------------------
// cmd = "/rs-mkfile/" | "/fs-mkfile/"

func Mkfile(
	c rpc.Client, ret interface{}, cmd, entry string,
	fsize int64, params, callbackParams string, ctxs []string) (code int, err error) {

	if callbackParams != "" {
		params += "/params/" + rpc.EncodeURI(callbackParams)
	}

	body := bytes.NewBuffer(make([]byte, 0, 176*len(ctxs)))
	for _, ctx := range ctxs {
		if _, err2 := body.WriteString(ctx + ","); err2 != nil {
			code, err = 400, errors.Info(errors.EINVAL, "Mkfile:", cmd, entry).Detail(err2)
			return
		}
	}
	if body.Len() > 0 {
		body.Truncate(body.Len() - 1)
	}

	code, err = c.CallWithBinaryEx(
		ret, UP_HOST+cmd+rpc.EncodeURI(entry)+"/fsize/"+strconv.FormatInt(fsize, 10)+params,
		"text/plain", body, body.Len())
	return
}

// ----------------------------------------------------------
// /mkfile/<Fsize>/mimeType/<EncodedMimeType>/x:user-var/<EncodedUserVarVal>/key/<EncodedKey>

func Mkfile2(
	c rpc.Client, ret interface{}, key, mimeType string,
	fsize int64, params map[string]string, ctxs []string) (code int, err error) {

	url := UP_HOST + "/mkfile/" + strconv.FormatInt(fsize, 10)
	if mimeType != "" {
		url += "/mimeType/" + base64.URLEncoding.EncodeToString([]byte(mimeType))
	}
	for k, v := range params {
		if strings.HasPrefix(k, "x:") && v != "" {
			url += "/" + k + "/" + base64.URLEncoding.EncodeToString([]byte(v))
		}
	}
	// when key is last one, key can be empty.
	if key != "?" {
		url += "/key/" + base64.URLEncoding.EncodeToString([]byte(key))
	}
	log.Info("mkfile2:", url)

	body := new(bytes.Buffer)
	for i, ctx := range ctxs {
		if i != len(ctxs)-1 {
			body.WriteString(ctx + ",")
		} else {
			body.WriteString(ctx)
		}
	}
	code, err = c.CallWithBinaryEx(ret, url, "application/octet-stream", body, body.Len())
	return
}

// ----------------------------------------------------------

// /mkfile/<Fsize>/mimeType/<EncodedMimeType>/fname/<EncodedFname>/x:user-var/<EncodedUserVarVal>/key/<EncodedKey>

func Mkfile3(
	c rpc.Client, ret interface{}, key, mimeType, fname string,
	fsize int64, params map[string]string, ctxs []string) (code int, err error) {

	url := UP_HOST + "/mkfile/" + strconv.FormatInt(fsize, 10)
	if mimeType != "" {
		url += "/mimeType/" + base64.URLEncoding.EncodeToString([]byte(mimeType))
	}
	if fname != "" {
		url += "/fname/" + base64.URLEncoding.EncodeToString([]byte(fname))
	}

	for k, v := range params {
		if strings.HasPrefix(k, "x:") && v != "" {
			url += "/" + k + "/" + base64.URLEncoding.EncodeToString([]byte(v))
		}
	}
	// when key is last one, key can be empty.
	if key != "?" {
		url += "/key/" + base64.URLEncoding.EncodeToString([]byte(key))
	}
	log.Info("mkfile3:", url)

	body := new(bytes.Buffer)
	for i, ctx := range ctxs {
		if i != len(ctxs)-1 {
			body.WriteString(ctx + ",")
		} else {
			body.WriteString(ctx)
		}
	}
	code, err = c.CallWithBinaryEx(ret, url, "application/octet-stream", body, body.Len())
	return
}

// ----------------------------------------------------------

func GlbMkfile(
	upHost string, c rpc.Client, ret interface{}, key, mimeType, fname string,
	fsize int64, params map[string]string, ctxs []string) (code int, err error) {

	if upHost == "" && fsize == 0 {
		upHost = GLB_UP_HOST
	}
	url := upHost + "/glb/mkfile/" + strconv.FormatInt(fsize, 10)
	if mimeType != "" {
		url += "/mimeType/" + base64.URLEncoding.EncodeToString([]byte(mimeType))
	}
	if fname != "" {
		url += "/fname/" + base64.URLEncoding.EncodeToString([]byte(fname))
	}

	for k, v := range params {
		if strings.HasPrefix(k, "x:") && v != "" {
			url += "/" + k + "/" + base64.URLEncoding.EncodeToString([]byte(v))
		}
	}
	// when key is last one, key can be empty.
	if key != "?" {
		url += "/key/" + base64.URLEncoding.EncodeToString([]byte(key))
	}
	log.Info("glb mkfile:", url)

	body := new(bytes.Buffer)
	for i, ctx := range ctxs {
		if i != len(ctxs)-1 {
			body.WriteString(ctx + ",")
		} else {
			body.WriteString(ctx)
		}
	}
	code, err = c.CallWithBinaryEx(ret, url, "application/octet-stream", body, body.Len())
	return
}

// ----------------------------------------------------------

type Service struct {
	Tasks chan func()
}

func New(taskQsize, threadSize int) Service {
	tasks := make(chan func(), taskQsize)
	for i := 0; i < threadSize; i++ {
		go worker(tasks)
	}
	return Service{tasks}
}

func worker(tasks chan func()) {
	for {
		task := <-tasks
		task()
	}
}

// ----------------------------------------------------------

func BlockCount(fsize int64) int {

	blockMask := int64((1 << BLOCK_BITS) - 1)
	return int((fsize + blockMask) >> BLOCK_BITS)
}

func (r Service) Put(
	c rpc.Client, f io.ReaderAt, fsize int64, checksums []string, progs []BlockputProgress,
	blockNotify func(blockIdx int, checksum string),
	chunkNotify func(blockIdx int, prog *BlockputProgress)) (code int, err error) {

	blockCnt := BlockCount(fsize)
	if len(checksums) != blockCnt || len(progs) != blockCnt {
		code, err = 400, errors.Info(errors.EINVAL, "up.Service.Put")
		return
	}

	var wg sync.WaitGroup
	wg.Add(blockCnt)

	last := blockCnt - 1
	blockSize := 1 << BLOCK_BITS

	var failed bool
	for i := 0; i < blockCnt; i++ {
		if checksums[i] == "" {
			blockIdx := i
			blockSize1 := blockSize
			if i == last {
				offbase := int64(blockIdx) << BLOCK_BITS
				blockSize1 = int(fsize - offbase)
			}
			task := func() {
				defer wg.Done()
				retry := PUT_RETRY_TIMES
			lzRetry:
				ret, code2, err2 := ResumableBlockput(
					c, f, blockIdx, blockSize1, PUT_CHUNK_SIZE, PUT_RETRY_TIMES, &progs[blockIdx], chunkNotify)
				if err2 != nil {
					if retry > 0 {
						retry--
						log.Info("ResumableBlockput retrying ...", err2)
						goto lzRetry
					}
					log.Warn("ResumableBlockput", blockIdx, "failed:", code2, errors.Detail(err2))
					failed = true
				} else {
					checksums[blockIdx] = ret.Checksum
					blockNotify(blockIdx, ret.Checksum)
				}
				progs[blockIdx].Err = err2
			}
			r.Tasks <- task
		} else {
			wg.Done()
		}
	}

	wg.Wait()
	if failed {
		code, err = api.FunctionFail, errors.Info(api.EFunctionFail, "up.Service.Put")
	} else {
		code = 200
	}
	return
}

func (r Service) GlbPut(
	uphost string, c rpc.Client, f io.ReaderAt, fsize int64, checksums []string, progs []BlockputProgress,
	blockNotify func(blockIdx int, checksum string),
	chunkNotify func(blockIdx int, prog *BlockputProgress)) (ret PutRet, code int, err error) {

	blockCnt := BlockCount(fsize)
	if len(checksums) != blockCnt || len(progs) != blockCnt {
		code, err = 400, errors.Info(errors.EINVAL, "up.Service.Put")
		return
	}

	var wg sync.WaitGroup
	var failed bool
	wg.Add(blockCnt)

	last := blockCnt - 1
	blockSize := 1 << BLOCK_BITS

	hlock := hostLock{host: uphost, needLock: uphost == GLB_UP_HOST}
	for i := 0; i < blockCnt; i++ {
		if checksums[i] == "" {
			blockIdx := i
			blockSize1 := blockSize
			if i == last {
				offbase := int64(blockIdx) << BLOCK_BITS
				blockSize1 = int(fsize - offbase)
			}
			if i == 0 && hlock.needLock {
				hlock.Lock()
			}
			task := func() {
				defer wg.Done()
				retry := PUT_RETRY_TIMES
			lzRetry:
				ret2, code2, err2 := GlbResumableBlockput(
					&hlock, c, f, blockIdx, blockSize1, PUT_CHUNK_SIZE, PUT_RETRY_TIMES, &progs[blockIdx], chunkNotify)
				if err2 != nil {
					if retry > 0 {
						retry--
						log.Info("GlbResumableBlockput retrying ...", err2)
						goto lzRetry
					}
					if blockIdx == 0 && hlock.needLock {
						hlock.err = err2
						hlock.Unlock()
					}
					log.Warn("GlbResumableBlockput", blockIdx, "failed:", code2, errors.Detail(err2))
					failed = true
				} else {
					ret = ret2
					checksums[blockIdx] = ret.Checksum
					blockNotify(blockIdx, ret.Checksum)
				}
				progs[blockIdx].Err = err2
			}
			r.Tasks <- task
		} else {
			wg.Done()
		}
	}

	wg.Wait()
	if failed {
		code, err = api.FunctionFail, errors.Info(api.EFunctionFail, "up.Service.Put")
	} else {
		code = 200
	}
	return
}

// ----------------------------------------------------------

type CallbackEnv struct {
	Url      string `json:"callback_url"`
	Host     string `json:"callback_host,omitempty"`
	BodyType string `json:"callback_bodyType"`
	Body     string `json:"callback_body"`
	FetchKey uint16 `json:"callback_fetchKey,omitempty"`
	Token    string `json:"token"`
	ErrCode  int    `json:"err_code"`
	ErrStr   string `json:"error"`
	Hash     string `json:"hash"`          //保持错误码不变，但增加 key 、 hash 的内容
	Key      string `json:"key,omitempty"` //保持错误码不变，但增加 key 、 hash 的内容 https://pm.qbox.me/issues/21643
}

func (e *CallbackEnv) Error() string {
	b, _ := json.Marshal(e)
	return string(b)
}

// ----------------------------------------------------------
