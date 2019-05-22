package mime

import (
	"bytes"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFixMimeType(t *testing.T) {
	cases := []struct {
		oldMime, fname, key, newMime string
	}{
		{"image/jpeg", "a.js", "a.css", "image/jpeg"},
		{"audio/x-vorbis+ogg", "a.ogg", "", "application/ogg"},
		{"audio/x-vorbis+ogg", "", "a.ogg", "application/ogg"},
		{"audio/midi", "a.js", "a.css", "audio/midi"},
		{"video/mp4", "a.js", "a.css", "video/mp4"},
		{"text/plain", "a.mp3", "a.mp3", "text/plain"},
		{"text/x-csrc", "a.js", "a.js", "text/javascript"},
		{"text/x-csrc", "a.css", "a.css", "text/css"},
		{"text/plain", "a.css", "a.css", "text/css"},
		{"text/plain", "a.CSS", "a.CSS", "text/css"},
		{"text/plain", "a.ipa", "a.mp3", "application/octet-stream"},
		{"text/plain", "a.mp3", "a.ipa", "application/octet-stream"},
		{"application/octet-stream", "a.css", "a.css", "text/css"},
		{"application/zip", "a.apk", "a.apk", "application/vnd.android.package-archive"},
		{"application/x-zip-compressed", "a.apk", "a.apk", "application/vnd.android.package-archive"},
		{"", "a.mp3", "a.mp3", "application/octet-stream"},
		{"audio/x-riff", "a.webp", "a.webp", "image/webp"},
		{"application/x-ole-storage", "a.xls", "a.xls", "application/vnd.ms-excel"},
		{"application/zip", "a.docx1234", "a.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
	}
	for _, v := range cases {
		assert.Equal(t, v.newMime, FixMimeType(v.fname, v.key, v.oldMime))
	}
	for k, v := range suffixMime {
		assert.Equal(t, v, FixMimeType("a"+k, "", "text/css"))
	}
	for k, v := range suffixMime {
		assert.Equal(t, v, FixMimeType("", "a"+k, "text/css"))
	}
	for k, v := range mimeSuffixMimeType {
		assert.Equal(t, v, FixMimeType("", "a"+k.Suffix, k.MimeType))
	}
	for k, v := range mimeSuffixMimeType {
		assert.Equal(t, v, FixMimeType("a"+k.Suffix, "", k.MimeType))
	}
	assert.Equal(t, "text/css", FixMimeType("a.jar", "b.jar", "text/css"))
	assert.Equal(t, "application/octet-stream", FixMimeType("a.jar", "b.jar", ""))
}

type bufferReadCloser struct {
	*bytes.Buffer
}

func (rc bufferReadCloser) Close() error {
	return nil
}
func TestDetectMimeTypeEmptyFile(t *testing.T) {
	var b []byte

	// io.ReadCloser
	var rc io.ReadCloser = &bufferReadCloser{bytes.NewBuffer(b)}
	mime, err := ReadMimeType(&rc)
	assert.Nil(t, err)
	assert.Empty(t, mime)

	// io.ReadSeek
	rs := bytes.NewReader(b)
	mime, err = MimeType(rs)
	assert.Nil(t, err)
	assert.Empty(t, mime)
}
