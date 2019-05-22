package db

import (
	"github.com/qiniu/db/mgoutil.v3"
)

const (
	DefaultCollSessionPoolLimit = 100
)

type Collections struct {
	Alias            mgoutil.Collection `coll:"alias"`
	FaceClusterTasks mgoutil.Collection `coll:"face_cluster_tasks"`
	FaceFeatureTasks mgoutil.Collection `coll:"face_feature_tasks"`
	Faces            mgoutil.Collection `coll:"faces"`
	Groups           mgoutil.Collection `coll:"groups"`
	Images           mgoutil.Collection `coll:"images"`
	DataVersions     mgoutil.Collection `coll:"data_versions"`
}

var collections *Collections

func Init(cfg *mgoutil.Config) error {

	colls := &Collections{}
	mgoSession, err := mgoutil.Open(colls, cfg)
	if err != nil {
		return err
	}
	// NO CLOSE

	mgoSession.SetPoolLimit(DefaultCollSessionPoolLimit)

	collections = colls

	// Ensure Indexes
	{
		dao, _ := NewAliasDao()
		if err = dao.(*aliasDao).EnsureIndexes(); err != nil {
			return err
		}
	}
	{
		dao, _ := NewClusterTaskDao()
		if err = dao.(*clusterTaskDao).EnsureIndexes(); err != nil {
			return err
		}
	}
	{
		dao, _ := NewFeatureTaskDao()
		if err = dao.(*featureTaskDao).EnsureIndexes(); err != nil {
			return err
		}
	}
	{
		dao, _ := NewFaceDao()
		if err = dao.(*faceDao).EnsureIndexes(); err != nil {
			return err
		}
	}
	{
		dao, _ := NewGroupDao()
		if err = dao.(*groupDao).EnsureIndexes(); err != nil {
			return err
		}
	}
	{
		dao, _ := NewImageDao()
		if err = dao.(*imageDao).EnsureIndexes(); err != nil {
			return err
		}
	}

	return nil
}
