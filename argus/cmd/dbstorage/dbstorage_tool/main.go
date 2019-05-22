package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/cc/config"

	"qiniu.com/argus/dbstorage/dao"
	"qiniu.com/argus/dbstorage/job"
	outer "qiniu.com/argus/dbstorage/outer_service"
	"qiniu.com/argus/dbstorage/proto"
	"qiniu.com/argus/dbstorage/source"
	"qiniu.com/argus/dbstorage/util"
)

var (
	imageSourceFolderPath = "./source/"
	imageSourceFile       = "./urlSource"
	outputLogPath         = "./dbstorage_log/%s/log/"
	internalLogPath       = "./dbstorage_log/%s/processlog/"
	errorFilePath         = outputLogPath + "error"
	countFilePath         = outputLogPath + "count"
	processFilePath       = internalLogPath + "process"
	hashFilePath          = internalLogPath + "hash"
)

type Config struct {
	ImageFolderPath     string                  `json:"image_folder_path"`
	ImageListFile       string                  `json:"image_list_file"`
	LoadImageFromFolder bool                    `json:"load_image_from_folder"`
	ThreadNum           int                     `json:"thread_num"`
	GroupName           proto.GroupName         `json:"group_name"`
	FeatureGroupService proto.BaseServiceConfig `json:"feature_group_service"`
	ServingService      proto.BaseServiceConfig `json:"serving_service"`
	TaskConfig          proto.TaskConfig        `json:"task_config"`
	IsPrivate           bool                    `json:"is_private"`
}

func main() {
	var (
		ctx     = xlog.NewContext(context.Background(), xlog.NewDummy())
		xl      = xlog.FromContextSafe(ctx)
		conf    Config
		src     source.ISource
		count   int
		srcPath string
	)

	runtime.GOMAXPROCS(runtime.NumCPU())

	//load config file
	config.Init("f", "dbstorage_tool", "dbstorage_tool.conf")
	if err := config.Load(&conf); err != nil {
		xl.Fatal("fail to load configure file")
	}

	if conf.ImageFolderPath != "" {
		imageSourceFolderPath = strings.TrimSuffix(conf.ImageFolderPath, "/") + "/"
	}
	if conf.ImageListFile != "" {
		imageSourceFile = conf.ImageListFile
	}

	//init source
	if conf.LoadImageFromFolder {
		srcPath = imageSourceFolderPath
		src = source.NewFolderSource(imageSourceFolderPath)
	} else {
		srcPath = imageSourceFile
		ext := strings.ToLower(filepath.Ext(imageSourceFile))
		if ext != ".csv" && ext != ".json" {
			xl.Fatal("only support csv and json file, file name must have extension .csv or .json !!!")
		}
		content, err := ioutil.ReadFile(imageSourceFile)
		if err != nil {
			xl.Fatalf("fail to read source file: %s", err.Error())
		}

		switch ext {
		case ".csv":
			src = source.NewCsvSource(content)
		case ".json":
			src = source.NewJsonSource(content)
		default:
			xl.Fatal("only support csv and json file, file name must have extension .csv or .json !!!")
		}

		count, err = src.GetInfo(ctx)
		if err != nil {
			xl.Fatal(err.Error())
		}
	}

	generatePath(srcPath)

	//create log path
	if err := util.CreatePath(outputLogPath); err != nil {
		xl.Fatalf("fail to create path: %s", err)
	}
	if err := util.CreatePath(internalLogPath); err != nil {
		xl.Fatalf("fail to create path: %s", err)
	}

	//init dao
	dao, err := dao.NewFileDao(errorFilePath, hashFilePath, countFilePath, processFilePath)
	if err != nil {
		xl.Fatalf("fail to call dao.NewFileDao: %s", err.Error())
	}

	//init task
	task := &proto.Task{
		GroupName:  conf.GroupName,
		Config:     conf.TaskConfig,
		TotalCount: count,
	}

	//init config
	config := proto.TaskServiceConfig{
		FeatureGroupService: conf.FeatureGroupService,
		ServingService:      conf.ServingService,
		ThreadNum:           conf.ThreadNum,
	}

	//init face group service
	outerService := outer.NewFaceGroup(
		conf.FeatureGroupService.Host,
		conf.FeatureGroupService.Timeout*time.Second,
		conf.IsPrivate,
		1, 0,
	)

	//init job dispatcher
	dispatcher := job.NewDispatcher(dao, &config)

	//start
	if err := dispatcher.New(ctx, task, src, outerService, false, true); err != nil {
		xl.Fatalf("fail to start service: %s", err.Error())
	}
}

func generatePath(path string) {
	path = strings.Replace(path, "/", ":", -1)
	outputLogPath = fmt.Sprintf(outputLogPath, path)
	internalLogPath = fmt.Sprintf(internalLogPath, path)
	errorFilePath = fmt.Sprintf(errorFilePath, path)
	countFilePath = fmt.Sprintf(countFilePath, path)
	processFilePath = fmt.Sprintf(processFilePath, path)
	hashFilePath = fmt.Sprintf(hashFilePath, path)
}
