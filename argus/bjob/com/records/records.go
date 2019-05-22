package records

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
	// "github.com/seiflotfy/cuckoofilter"
	"qiniu.com/argus/com/3rd/github.com/seiflotfy/cuckoofilter"

	httputil "qiniupkg.com/http/httputil.v2"
)

type RecordKey string
type RecordValue []byte

type Records interface {
	Count(context.Context) (int, error)
	HasKey(context.Context, RecordKey) (bool, error)
	Append(context.Context, RecordKey, RecordValue) error
	Close(context.Context) ([]string, error)
}

////////////////////////////////////////////////////////////////////////////////

type RecordStorage interface {
	Scan(context.Context) _RecordStorageScanner
	Append(context.Context, RecordKey, RecordValue) error
	Flush(context.Context) (string, error)
}

type _RecordStorageScanner interface {
	Scan(context.Context) (RecordKey, RecordValue, error)
	Close(context.Context) error
}

type StorageList interface {
	List(context.Context) ([]string, error)
}

////////////////////////////////////////////////////////////////////////////////

var _ Records = &records{}

type records struct {
	FileKeys []string
	RecordStorage
	*cuckoofilter.CuckooFilter
	*sync.Mutex
}

func NewRecords(ctx context.Context, stg RecordStorage, capacity uint) *records {

	xl := xlog.FromContextSafe(ctx)
	var cf *cuckoofilter.CuckooFilter
	var realCapacity uint
	fkeys := make([]string, 0)
	if capacity > 0 {
		cf, realCapacity = cuckoofilter.NewCuckooFilter(capacity)

		scan := stg.Scan(ctx)
		defer scan.Close(ctx)

		var count int64 = 0
		for {
			linekey, _, err := scan.Scan(ctx)
			if err != nil {
				if err == io.EOF {
					err = nil
				} else {
					xl.Errorf("CuckooFilter load err, %v", err)
				}
				break
			}

			cf.InsertUnique([]byte(linekey))

			count++
			if count%100000 == 0 {
				xl.Infof("load records. %d", count)
			}

		}

		if lst, ok := scan.(StorageList); ok {
			fkeys, _ = lst.List(ctx)
		}
		xl.Infof("CuckooFilter capacity %d , load %d", int(realCapacity), cf.Count())
	}
	xl.Infof("FileKeys load %+v", fkeys)

	return &records{
		FileKeys:      fkeys,
		RecordStorage: stg,
		CuckooFilter:  cf,
		Mutex:         new(sync.Mutex),
	}
}

func (rs *records) Count(context.Context) (int, error) {
	if rs.CuckooFilter == nil {
		return -1, nil
	}
	return int(rs.CuckooFilter.Count()), nil
}
func (rs *records) HasKey(ctx context.Context, key RecordKey) (bool, error) {
	if rs.CuckooFilter == nil {
		return false, nil
	}
	return rs.CuckooFilter.Lookup([]byte(key)), nil
}
func (rs *records) Append(ctx context.Context, key RecordKey, value RecordValue) error {
	rs.Lock()
	defer rs.Unlock()
	if rs.CuckooFilter == nil || rs.CuckooFilter.InsertUnique([]byte(key)) {
		return rs.RecordStorage.Append(ctx, key, value)
	}
	return nil
}
func (rs *records) Close(ctx context.Context) ([]string, error) {
	xl := xlog.FromContextSafe(ctx)

	var err error
	key := ""
	retryDuration := time.Millisecond * 200
	for i := 0; i < 5; i++ {
		key, err = rs.RecordStorage.Flush(ctx)
		if err == nil {
			func() {
				rs.Lock()
				defer rs.Unlock()
				rs.FileKeys = append(rs.FileKeys, key)
			}()
			break
		}

		// print err
		ec := httputil.DetectCode(err)
		xl.Errorf("Flush err, %d, %+v, retry = %d", ec, err, i)

		time.Sleep(retryDuration)
		retryDuration *= 2
	}

	if lst, ok := rs.RecordStorage.(StorageList); ok {
		keys, _ := lst.List(ctx)
		rs.FileKeys = append(rs.FileKeys, keys...)
	}

	return rs.FileKeys, err
}
