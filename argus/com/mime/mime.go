package mime

import (
	"bytes"
	"io"
	"path"
	"strings"

	"bitbucket.org/taruti/mimemagic"
)

const (
	mimemagicBufSize = 1024
)

// =======================================================================

func MimeType(f io.ReadSeeker) (mimeType string, err error) {

	buf := make([]byte, mimemagicBufSize)
	n, err := io.ReadFull(f, buf[:])
	if n != mimemagicBufSize {
		if err != io.ErrUnexpectedEOF && err != io.EOF {
			return
		}
		err = nil
	}
	mimeType = mimemagic.Match("", buf[:n])
	_, err = f.Seek(0, io.SeekStart)
	return
}

/*
   根据内容检测不一定准确，再根据文件名和key进行修正。
   修正策略：
   1. 音视频，图片这类富媒体根据内容检测出来是比较靠谱的，不进行修正
   2. 其他常见类型直接根据后缀名进行修正，见suffixMime
   注意：
   1. 针对新文件类型，比如apk，处理的话还需要在 base/com/src/qbox.us/cc/mime/mime.go 进行注册
*/

var suffixMime = map[string]string{
	".css":      "text/css",
	".js":       "text/javascript",
	".txt":      "text/plain",
	".text":     "text/plain",
	".htm":      "text/html",
	".html":     "text/html",
	".shtml":    "text/html",
	".manifest": "text/cache-manifest",
	".wml":      "text/vnd.wap.wml",
	".wmls":     "text/vnd.wap.wmlscript",
	".go":       "text/x-go",
	".py":       "text/x-python",
	".rb":       "text/x-ruby",
	".java":     "text/x-java",
	".c":        "text/x-csrc",
	".hs":       "text/x-haskell",
	".cpp":      "text/x-c++src",
	".c++":      "text/x-c++src",
	".cc":       "text/x-c++src",
	".csv":      "text/csv",
	".ics":      "text/calendar",
	".json":     "application/json",
	".atom":     "application/atom+xml",
	".rss":      "application/rss+xml",
	".php":      "application/x-httpd-php",
	".torrent":  "application/x-bittorrent",
	".sis":      "application/vnd.symbian.install",
	".xml":      "application/xml",
	".xhtml":    "application/xhtml+xml",
	".xht":      "application/xhtml+xml",
	".ipa":      "application/octet-stream",
}

type fixMimeCond struct {
	MimeType string
	Suffix   string
}

var mimeSuffixMimeType = map[fixMimeCond]string{
	fixMimeCond{"application/x-ole-storage", ".doc"}:    "application/msword",
	fixMimeCond{"application/x-ole-storage", ".ppt"}:    "application/vnd.ms-powerpoint",
	fixMimeCond{"application/x-ole-storage", ".pps"}:    "application/vnd.ms-powerpoint",
	fixMimeCond{"application/x-ole-storage", ".xls"}:    "application/vnd.ms-excel",
	fixMimeCond{"application/zip", ".docx"}:             "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	fixMimeCond{"application/zip", ".pptx"}:             "application/vnd.openxmlformats-officedocument.presentationml.presentation",
	fixMimeCond{"application/zip", ".ppsx"}:             "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
	fixMimeCond{"application/zip", ".xlsx"}:             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	fixMimeCond{"application/zip", ".apk"}:              "application/vnd.android.package-archive",
	fixMimeCond{"application/x-zip-compressed", ".apk"}: "application/vnd.android.package-archive",
	fixMimeCond{"audio/x-riff", ".webp"}:                "image/webp",
}

func fixMimeByCond(mimeType, suffix string) (mimeTypeNew string) {
	cond := fixMimeCond{
		MimeType: mimeType,
		Suffix:   suffix,
	}
	return mimeSuffixMimeType[cond]
}

func FixMimeType(fName, key, mimeType string) (mimeTypeNew string) {
	if strings.HasPrefix(mimeType, "audio/") && (path.Ext(fName) == ".ogg" || path.Ext(key) == ".ogg") {
		mimeTypeNew = "application/ogg"
		return
	}

	fName = strings.ToLower(fName)
	key = strings.ToLower(key)
	if fmime := fixMimeByCond(mimeType, path.Ext(fName)); fmime != "" {
		return fmime
	}
	if kmime := fixMimeByCond(mimeType, path.Ext(key)); kmime != "" {
		return kmime
	}

	if strings.HasPrefix(mimeType, "audio/") || strings.HasPrefix(mimeType, "image/") ||
		strings.HasPrefix(mimeType, "video/") {
		mimeTypeNew = mimeType
		return
	}

	if fmime := suffixMime[path.Ext(fName)]; fmime != "" {
		mimeTypeNew = fmime
		return
	}
	if kmime := suffixMime[path.Ext(key)]; kmime != "" {
		mimeTypeNew = kmime
		return
	}

	if mimeType == "" {
		mimeTypeNew = "application/octet-stream"
		return
	}
	mimeTypeNew = mimeType
	return
}

func MimeType2(fName string, key string, f io.ReadSeeker) (mimeType string) {
	mimeType, _ = MimeType(f)
	return FixMimeType(fName, key, mimeType)
}

// =======================================================================

type readCloser struct {
	io.Reader
	io.Closer
}

func ReadMimeType(rc *io.ReadCloser) (mimeType string, err error) {

	r := *rc
	if f, ok := r.(io.ReadSeeker); ok {
		return MimeType(f)
	}

	buf := make([]byte, mimemagicBufSize)
	n, err := io.ReadFull(r, buf)
	if n != mimemagicBufSize {
		if err != io.ErrUnexpectedEOF && err != io.EOF {
			return
		}
		err = nil
	}
	mimeType = mimemagic.Match("", buf[:n])
	mr := io.MultiReader(bytes.NewReader(buf[:n]), r)
	*rc = &readCloser{mr, r}
	return
}
