package main

import (
	"fmt"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

const (
	defaultCollSessionPoolLimit = 100
)

type Storage struct {
	Collections mgoutil.Collection `coll:"ffmpeg_infos"`
}

func NewStorage(cfg mgoutil.Config) (*Storage, error) {
	storage := Storage{}
	mgoSession, err := mgoutil.Open(&storage, &cfg)
	if err != nil {
		return nil, err
	}

	mgoSession.SetPoolLimit(defaultCollSessionPoolLimit)

	// ensure index
	if err = storage.Collections.EnsureIndex(mgo.Index{Key: []string{"camera_id"}, Unique: true}); err != nil {
		return nil, fmt.Errorf("groups collections ensure index name err: %s", err.Error())
	}

	return &storage, nil
}

func (s *Storage) Get(id string) (ffmpegInfo FfmpegInfo, exist bool, err error) {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	if err = col.Find(bson.M{"camera_id": id}).One(&ffmpegInfo); err != nil {
		exist = false
		if err == mgo.ErrNotFound {
			err = nil
			return
		}
		return
	}
	exist = true
	return
}

func (s *Storage) Add(ffmpegInfo FfmpegInfo) error {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	if err := col.Insert(ffmpegInfo); err != nil {
		return err
	}
	return nil
}

func (s *Storage) Update(id string, ffmpegInfo FfmpegInfo) error {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	if err := col.Update(bson.M{"camera_id": id}, ffmpegInfo); err != nil {
		return err
	}
	return nil
}

func (s *Storage) Remove(id string) error {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	if err := col.Remove(bson.M{"camera_id": id}); err != nil {
		return err
	}
	return nil
}

func (s *Storage) All() ([]FfmpegInfo, error) {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	ffmpegInfos := []FfmpegInfo{}
	if err := col.Find(nil).All(&ffmpegInfos); err != nil {
		return ffmpegInfos, err
	}
	return ffmpegInfos, nil
}

func (s *Storage) Destroy() error {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	if _, err := col.RemoveAll(nil); err != nil {
		return err
	}
	return nil
}

func (s *Storage) Count() (n int, err error) {
	col := s.Collections.CopySession()
	defer col.CloseSession()
	if n, err = col.Count(); err != nil {
		return
	}
	return
}
