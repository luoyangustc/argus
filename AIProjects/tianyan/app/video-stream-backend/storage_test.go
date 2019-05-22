package main

import (
	"math/rand"
	"testing"
	"time"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/suite"
)

const (
	_MGO_HOST = "localhost"
	_MGO_DB   = "test"
)

type StorageTestSuite struct {
	suite.Suite
	storage *Storage
}

func (s *StorageTestSuite) SetupTest() {
	s.storage, _ = NewStorage(mgoutil.Config{
		Host: _MGO_HOST,
		DB:   _MGO_DB,
		Mode: "strong",
	})
}

func (s *StorageTestSuite) TearDownSuite() {
	s.storage.Destroy()
}

func (s *StorageTestSuite) TestStorage() {
	cameraID := randSeq(10)

	s.Nil(s.storage.Add(FfmpegInfo{
		CameraID: cameraID,
		Pid:      1,
	}))

	n, err := s.storage.Count()
	s.Nil(err)
	s.Equal(1, n)

	ffs, err := s.storage.All()
	s.Nil(err)
	s.Equal(1, len(ffs))

	_, exist, err := s.storage.Get(cameraID + "1")
	s.Nil(err)
	s.False(exist)

	ff, exist, err := s.storage.Get(cameraID)
	s.Nil(err)
	s.True(exist)
	s.Equal(1, ff.Pid)

	s.Nil(s.storage.Update(cameraID, FfmpegInfo{
		CameraID: cameraID,
		Pid:      2,
	}))

	ff, exist, err = s.storage.Get(cameraID)
	s.Nil(err)
	s.True(exist)
	s.Equal(2, ff.Pid)

	s.Nil(s.storage.Remove(cameraID))

	n, err = s.storage.Count()
	s.Nil(err)
	s.Equal(0, n)
}

func TestStorageTestSuite(t *testing.T) {
	suite.Run(t, new(StorageTestSuite))
}

var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

func randSeq(n int) string {
	rand.Seed(time.Now().UnixNano())
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}
