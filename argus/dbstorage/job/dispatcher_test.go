package job

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/dbstorage/proto"
)

type MockDao struct {
	Log proto.Log
}

func (d *MockDao) NewTask(ctx context.Context, id proto.TaskId, group proto.GroupName, config proto.TaskConfig, len int, ext string, uid uint32) (err error) {
	return
}
func (d *MockDao) DeleteTask(ctx context.Context, id proto.TaskId, uid uint32) (err error) { return }
func (d *MockDao) GetTask(ctx context.Context, id proto.TaskId, uid uint32) (task *proto.Task, err error) {
	return
}
func (d *MockDao) GetTaskList(ctx context.Context, group proto.GroupName, status proto.TaskStatus, uid uint32) (ids []proto.Task, err error) {
	return
}
func (d *MockDao) ResetTask(ctx context.Context, stat proto.TaskStatus, status ...proto.TaskStatus) (err error) {
	return
}
func (d *MockDao) UpdateTaskStatus(ctx context.Context, id proto.TaskId, status proto.TaskStatus) (err error) {
	return
}
func (d *MockDao) UpdateTaskCount(ctx context.Context, id proto.TaskId, count int) (err error) {
	return
}
func (d *MockDao) UpdateTaskError(ctx context.Context, id proto.TaskId, taskErr proto.TaskError) (err error) {
	return
}
func (d *MockDao) NewTaskFile(ctx context.Context, name string, content []byte) (err error) { return }
func (d *MockDao) DeleteTaskFile(ctx context.Context, name string) (err error)              { return }
func (d *MockDao) GetTaskFile(ctx context.Context, name string) (content []byte, err error) { return }
func (d *MockDao) NewTaskLog(ctx context.Context, id proto.TaskId, uid uint32) (err error)  { return }
func (d *MockDao) GetTaskLog(ctx context.Context, id proto.TaskId) (*proto.Log, error) {
	return &d.Log, nil
}
func (d *MockDao) DeleteTaskLog(ctx context.Context, id proto.TaskId) (err error) { return }
func (d *MockDao) UpdateProcess(ctx context.Context, id proto.TaskId, process []int) (err error) {
	return
}
func (d *MockDao) UpdateHash(ctx context.Context, id proto.TaskId, hash string) (err error) { return }
func (d *MockDao) UpdateErrorLog(ctx context.Context, id proto.TaskId, log proto.ErrorLog) (err error) {
	return
}
func (d *MockDao) GetErrorLog(ctx context.Context, id proto.TaskId, skip, limit int, uid uint32) (log []proto.ErrorLog, count int, err error) {
	return
}
func (d *MockDao) GetErrorCount(ctx context.Context, taskId proto.TaskId, uid uint32) (count int, err error) {
	return
}

type MockSource struct {
	uri     []string
	content [][]byte
}

func (s *MockSource) Read(ctx context.Context, fn func(int) proto.ImageProcess) (<-chan proto.TaskSource, error) {
	ch := make(chan proto.TaskSource)
	go func() {
		for i, v := range s.content {
			fn(i)
			ch <- proto.TaskSource{
				Index:   i,
				URI:     proto.ImageURI(v),
				Content: s.content[i],
			}
			time.Sleep(50 * time.Millisecond)
		}
		close(ch)
	}()

	return ch, nil
}

func (s *MockSource) GetInfo(ctx context.Context) (int, error) {
	return 0, nil
}

type MockFG struct {
}

func (s *MockFG) Add(ctx context.Context, config proto.TaskConfig, group proto.GroupName, id proto.ImageId, uri proto.ImageURI, tag proto.ImageTag, desc proto.ImageDesc) (string, error) {
	return "", nil
}
func (s *MockFG) CreateGroup(ctx context.Context, group string) error { return nil }

func Test_Dispatcher(t *testing.T) {
	ctx := context.Background()

	faceGroup := &MockFG{}
	dao := &MockDao{
		Log: proto.Log{
			TaskId:  "id1",
			Process: []int{0, 1},
			Hash:    []string{"hash"},
		},
	}
	task1 := &proto.Task{
		TaskId:     "id1",
		GroupName:  "group_name1",
		TotalCount: 3,
	}
	src1 := &MockSource{uri: []string{"uri1", "uri2"}, content: [][]byte{[]byte("content1"), []byte("content2")}}

	task2 := &proto.Task{
		TaskId:     "id2",
		GroupName:  "group_name2",
		TotalCount: 3,
	}
	src2 := &MockSource{uri: []string{"uri3", "ur4"}, content: [][]byte{[]byte("content3"), []byte("content3")}}

	task3 := &proto.Task{
		TaskId:     "id3",
		GroupName:  "group_name3",
		TotalCount: 3,
	}
	src3 := &MockSource{uri: []string{"uri5", "uri6"}, content: [][]byte{nil, nil}}

	task4 := &proto.Task{
		TaskId:     "id4",
		GroupName:  "group_name4",
		TotalCount: 3,
	}
	src4 := &MockSource{uri: []string{"uri7", "uri8"}, content: [][]byte{nil, nil}}

	d := NewDispatcher(dao, &proto.TaskServiceConfig{
		ThreadNum:          20,
		MaxParallelTaskNum: 2,
	})

	_ = d.New(ctx, task1, src1, faceGroup, true, true)
	_ = d.New(ctx, task2, src2, faceGroup, true, false)
	_ = d.New(ctx, task3, src3, faceGroup, true, false)
	_ = d.New(ctx, task4, src4, faceGroup, true, false)

	assert.Equal(t, 2, len(d.runningTasks))
	assert.Equal(t, 2, len(d.pendingTasks))
	assert.Equal(t, proto.TaskId("id3"), d.pendingTasks[0].task.TaskId)
	assert.Equal(t, proto.TaskId("id4"), d.pendingTasks[1].task.TaskId)

	_ = d.Stop(ctx, "id3")
	assert.Equal(t, 2, len(d.runningTasks))
	assert.Equal(t, 1, len(d.pendingTasks))
	assert.Equal(t, proto.TaskId("id4"), d.pendingTasks[0].task.TaskId)

	_ = d.Stop(ctx, "id2")
	time.Sleep(300 * time.Millisecond)
	d.Lock()
	assert.Equal(t, 0, len(d.runningTasks))
	assert.Equal(t, 0, len(d.pendingTasks))
	d.Unlock()
}
