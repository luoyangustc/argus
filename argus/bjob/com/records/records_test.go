package records

import (
	"context"
	"math"
	"os"
	"path"
	"testing"

	"github.com/qiniu/xlog.v1"
	// "github.com/seiflotfy/cuckoofilter"
	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/com/3rd/github.com/seiflotfy/cuckoofilter"

	"qiniu.com/argus/argus/com/bucket"
)

func BenchmarkRecords(b *testing.B) {

	cf, _ := cuckoofilter.NewDefaultCuckooFilter()
	for i := 0; i < b.N; i++ {
		// cf, _ := cuckoofilter.NewCuckooFilter(1024 * 1024 * 1024)
		for i1 := 0; i1 <= math.MaxUint8; i1++ {
			for i2 := 0; i2 <= math.MaxUint8; i2++ {
				// for i3 := 0; i3 <= math.MaxUint8; i3++ {
				// for i4 := 0; i4 <= math.MaxUint8/16; i4++ {
				// cf.InsertUnique([]byte{byte(i1), byte(i2), byte(i3), byte(i4)})
				// }
				// cf.InsertUnique([]byte{byte(i1), byte(i2), byte(i3)})
				// }
				cf.InsertUnique([]byte{byte(i1), byte(i2)})
			}
		}
	}

}

func TestRecords(t *testing.T) {

	dir := path.Join(os.TempDir(), xlog.GenReqId())
	os.MkdirAll(dir, 0755)
	defer os.RemoveAll(dir)
	file := NewFile(bucket.NewLocalFS(dir), 0, 0)
	rs := NewRecords(context.Background(), file, 1024)

	count, _ := rs.Count(context.Background())
	assert.Equal(t, 0, count)
	rs.Append(context.Background(), "AAAA", []byte("XXXX"))
	count, _ = rs.Count(context.Background())
	assert.Equal(t, 1, count)
	ok, _ := rs.HasKey(context.Background(), "AAAA")
	assert.True(t, ok)
	rs.Append(context.Background(), "AAAA", []byte("XXXX"))
	count, _ = rs.Count(context.Background())
	assert.Equal(t, 1, count)
	rs.Append(context.Background(), "AAAB", []byte("XXXX"))
	count, _ = rs.Count(context.Background())
	assert.Equal(t, 2, count)
	rs.Close(context.Background())

	rs = NewRecords(context.Background(), file, 1024)
	count, _ = rs.Count(context.Background())
	assert.Equal(t, 2, count)
	rs.Close(context.Background())

	rs = NewRecords(context.Background(), file, 0)
	count, _ = rs.Count(context.Background())
	assert.Equal(t, -1, count)
	ok, _ = rs.HasKey(context.Background(), "AAAA")
	assert.False(t, ok)
	rs.Close(context.Background())
}
