package dao

import (
	"context"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"qiniu.com/argus/dbstorage/proto"
)

var _ IDao = new(FileDao)

type FileDao struct {
	errorFile   *proto.SafeFile
	hashFile    *proto.SafeFile
	countFile   *proto.SafeFile
	processFile *proto.SafeFile
}

func NewFileDao(errorFilePath, hashFilePath, countFilePath, processFilePath string) (IDao, error) {
	errorFile, err := proto.NewSafeFile(errorFilePath, proto.APPEND)
	if err != nil {
		return nil, errors.Errorf("error when opening file [%s]: %v\n", errorFilePath, err)
	}

	hashFile, err := proto.NewSafeFile(hashFilePath, proto.APPEND)
	if err != nil {
		return nil, errors.Errorf("error when opening file [%s]: %v\n", hashFilePath, err)
	}

	countFile, err := proto.NewSafeFile(countFilePath, proto.REPLACE)
	if err != nil {
		return nil, errors.Errorf("error when opening file [%s]: %v\n", countFilePath, err)
	}

	processFile, err := proto.NewSafeFile(processFilePath, proto.REPLACE)
	if err != nil {
		return nil, errors.Errorf("error when opening file [%s]: %v\n", processFilePath, err)
	}

	d := &FileDao{
		errorFile:   errorFile,
		hashFile:    hashFile,
		countFile:   countFile,
		processFile: processFile,
	}
	return d, nil
}

func (d *FileDao) UpdateTaskCount(ctx context.Context, taskId proto.TaskId, count int) error {
	if err := d.countFile.Write(strconv.Itoa(count)); err != nil {
		return errors.Errorf("fail to update task count due to err: %s", err)
	}
	return nil
}

func (d *FileDao) GetTaskLog(ctx context.Context, taskId proto.TaskId) (*proto.Log, error) {
	hash, err := d.hashFile.ReadLines()
	if err != nil {
		return nil, errors.Errorf("fail to get task log due to read hash err: %s", err)
	}

	processContent, err := d.processFile.ReadAll()
	if err != nil {
		return nil, errors.Errorf("fail to get task log due to read process err: %s", err)
	}

	process := []int{}
	processStr := string(processContent)
	if processStr != "" {
		allProcess := strings.Split(string(processContent), ",")
		for _, s := range allProcess {
			i, err := strconv.Atoi(s)
			if err != nil {
				return nil, errors.Errorf("fail to get task log due to parse process err, content has non-numeric charactor: %s", err)
			}
			process = append(process, i)
		}
	}

	errLogs, err := d.errorFile.ReadLines()
	if err != nil {
		return nil, errors.Errorf("fail to get error when reading error file: %s", err)
	}

	return &proto.Log{
		Process:  process,
		Hash:     hash,
		ErrCount: len(errLogs),
	}, nil
}

func (d *FileDao) UpdateProcess(ctx context.Context, taskId proto.TaskId, process []int) error {
	var str []string
	for _, i := range process {
		str = append(str, strconv.Itoa(i))
	}
	if err := d.processFile.Write(strings.Join(str, ",")); err != nil {
		return errors.Errorf("fail to update task process due to err: %s", err)
	}
	return nil
}

func (d *FileDao) UpdateHash(ctx context.Context, taskId proto.TaskId, hash string) error {
	if err := d.hashFile.Write(hash); err != nil {
		return errors.Errorf("fail to update task hash due to err: %s", err)
	}
	return nil
}

func (d *FileDao) UpdateErrorLog(ctx context.Context, taskId proto.TaskId, errLog proto.ErrorLog) error {
	if err := d.errorFile.Write(errLog.String()); err != nil {
		return errors.Errorf("fail to update error log due to err: %s", err)
	}
	return nil
}

func (d *FileDao) NewTask(ctx context.Context, taskId proto.TaskId, groupName proto.GroupName, taskConfig proto.TaskConfig, count int, ext string, uid uint32) error {
	return nil
}

func (d *FileDao) DeleteTask(ctx context.Context, taskId proto.TaskId, uid uint32) error {
	return nil
}

func (d *FileDao) GetTask(ctx context.Context, taskId proto.TaskId, uid uint32) (*proto.Task, error) {
	return nil, nil
}

func (d *FileDao) GetTaskList(ctx context.Context, groupName proto.GroupName, status proto.TaskStatus, uid uint32) ([]proto.Task, error) {
	return nil, nil
}

func (d *FileDao) UpdateTaskStatus(ctx context.Context, taskId proto.TaskId, status proto.TaskStatus) error {
	return nil
}

func (d *FileDao) NewTaskFile(ctx context.Context, fileName string, fileContent []byte) error {
	return nil
}

func (d *FileDao) DeleteTaskFile(ctx context.Context, fileName string) error {
	return nil
}

func (d *FileDao) GetTaskFile(ctx context.Context, fileName string) ([]byte, error) {
	return nil, nil
}

func (d *FileDao) NewTaskLog(ctx context.Context, taskId proto.TaskId, uid uint32) error {
	return nil
}

func (d *FileDao) DeleteTaskLog(ctx context.Context, taskId proto.TaskId) error {
	return nil
}

func (d *FileDao) GetErrorLog(ctx context.Context, taskId proto.TaskId, skip, limit int, uid uint32) ([]proto.ErrorLog, int, error) {
	return nil, 0, nil
}

func (d *FileDao) UpdateTaskError(ctx context.Context, taskId proto.TaskId, err proto.TaskError) error {
	return nil
}

func (d *FileDao) ResetTask(context.Context, proto.TaskStatus, ...proto.TaskStatus) error {
	return nil
}

func (d *FileDao) GetErrorCount(ctx context.Context, taskId proto.TaskId, uid uint32) (int, error) {
	return 0, nil
}
