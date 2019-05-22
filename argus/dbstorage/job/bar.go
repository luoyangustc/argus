package job

import pb "qiniu.com/argus/dbstorage/pb"

func InitBar(count int) *pb.ProgressBar {
	bar := pb.StartNew(count)
	bar.ShowSpeed = true
	bar.ShowElapsedTime = true
	bar.ShowFinalTime = false
	return bar
}

func IncrementBar(bar *pb.ProgressBar, success bool) {
	if bar != nil {
		if success {
			bar.IncrementSuccess()
		} else {
			bar.IncrementFailure()
		}
	}
}

func CompleteBar(bar *pb.ProgressBar, msg string) {
	if bar != nil {
		bar.FinishPrint(msg)
	}
}
