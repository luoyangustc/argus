package job

type Worker struct {
	index   int
	jobPool chan *FaceJob
	stop    chan bool
}

func NewWorker(index int, jobPool chan *FaceJob) *Worker {
	return &Worker{
		index:   index,
		jobPool: jobPool,
		stop:    make(chan bool)}
}

func (w *Worker) Start() {
	go func() {
		for {
			select {
			case job, ok := <-w.jobPool:
				if !ok {
					return
				}
				job.execute(w.index)
			case <-w.stop:
				return
			}
		}
	}()
}

func (w *Worker) Stop() {
	go func() {
		w.stop <- true
	}()
}
