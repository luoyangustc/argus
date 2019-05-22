package dao

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"
)

const (
	BATCH_SIZE = 1000
)

func InsertEntries(
	ctx context.Context, setId string, mimeTypes []proto.MimeType, urls []string,
) int {

	var (
		xl         = xlog.FromContextSafe(ctx)
		entries    = make([]*proto.Entry, 0, 1000)
		count      = len(urls)
		i          int
		validCount int
	)

	for idx, v := range urls {
		if util.IsHttpUrl(v) {
			mimeType := proto.GetMimeTypeWithExt(v)
			if mimeType != proto.MimeTypeOther && mimeType.IsContained(mimeTypes) {
				entries = append(entries, &proto.Entry{
					SetId:    setId,
					Uri:      v,
					MimeType: mimeType,
				})
				i++
			}
		}

		if i == BATCH_SIZE || idx == (count-1) {
			// insert 1000 record one time
			err := EntryDao.BatchInsert(entries)
			if err != nil {
				xl.Errorf("entryDao.BatchInsert : %v", err)
			}

			// reset
			entries = entries[:0]
			validCount += i
			i = 0
		}
	}

	return validCount
}
