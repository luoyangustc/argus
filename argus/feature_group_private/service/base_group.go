package service

import (
	"context"
	"hash/crc32"
	"strings"
	"sync"
	"time"

	"qiniu.com/argus/com/sigmoid"
	"qiniu.com/argus/com/util"
	"qiniu.com/argus/feature_group_private/feature"

	"github.com/imdario/mergo"
	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/manager"
	database "qiniu.com/argus/feature_group_private/manager/mgo"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
	"qiniu.com/argus/feature_group_private/search/cpu"
	"qiniu.com/argus/feature_group_private/search/gpu"
)

const (
	defaultSearchLimit  = 100
	defaultInitTimeout  = 600
	defaultInitInterval = 1
)

type BaseGroupsConfig struct {
	MgoConfig            mgoutil.Config         `json:"mgo_config"`
	CollSessionPoolLimit int                    `json:"coll_session_pool_limit"`
	Sets                 search.Config          `json:"sets"`
	Mode                 string                 `json:"mode"`
	ClusterMode          bool                   `json:"cluster_mode"`
	ClusterSize          int                    `json:"cluster_size"`
	Address              proto.NodeAddress      `json:"address"`
	BaseFeatureTimeout   time.Duration          `json:"base_feature_timeout"`
	SigmoidConfig        *sigmoid.SigmoidConfig `json:"sigmoid"`
}

//------------------- BaseGroups -------------------//
var _ feature_group.IGroups = new(BaseGroups)

type BaseGroups struct {
	GroupsConfig BaseGroupsConfig
	Groups       manager.Groups
	Sets         search.Sets
	Nodes        []proto.Node
	baseFeature  feature.BaseFeature
}

func NewBaseGroups(ctx context.Context, config BaseGroupsConfig, prefix string) (*BaseGroups, error) {
	xl := xlog.FromContextSafe(ctx)
	groups, err := database.NewGroupsManager(&database.GroupsConfig{
		MgoConfig:            config.MgoConfig,
		CollSessionPoolLimit: config.CollSessionPoolLimit})
	if err != nil {
		xl.Errorf("NewBaseGroups database.NewGroupsManager error: %s", err)
		return nil, errors.Wrap(err, "database.NewGroupsManager")
	}
	var sets search.Sets
	if strings.ToUpper(config.Mode) == "GPU" {
		sets, err = gpu.NewSets(config.Sets)
	} else {
		sets, err = cpu.NewSets(config.Sets)
	}
	if err != nil {
		xl.Errorf("NewBaseGroups set.NewSets error: %s", err)
		return nil, errors.Wrap(err, "set.NewSets")
	}
	if sets == nil {
		xl.Error("NewBaseGroups set.NewSets failed to create sets, please check config")
		return nil, errors.New("invalid cpu/gpu mode")
	}
	s := &BaseGroups{
		GroupsConfig: config,
		Groups:       groups,
		Sets:         sets,
		baseFeature: feature.NewBaseFeature(
			config.BaseFeatureTimeout*time.Second,
			prefix),
	}
	start := time.Now()
	cnt, err := s.initGroups(ctx)
	if err != nil {
		s.Groups.UpsertNode(ctx, proto.Node{Address: config.Address, Capacity: 0, State: proto.NodeStateError})
		xl.Errorf("NewBaseGroups s.initGroups error: %s", err)
		return nil, errors.Wrap(err, "initGroups")
	}
	xl.Infof("load data %v , use time %v", cnt, time.Since(start))
	if !config.ClusterMode {
		return s, nil
	}
	intiWait := 0
	var needEnsureHashKey bool
	for intiWait <= defaultInitTimeout {
		allInitialized := true
		if s.Nodes, err = s.Groups.AllNodes(ctx); err != nil {
			s.Groups.UpsertNode(ctx, proto.Node{Address: config.Address, Capacity: 0, State: proto.NodeStateError})
			xl.Errorf("Groups.AllNodes error: %s", err)
			return s, err
		}
		for _, node := range s.Nodes {
			if !needEnsureHashKey && node.State == proto.NodeEnsureHashKey {
				needEnsureHashKey = true
				cnt, err := s.initGroups(ctx)
				if err != nil {
					s.Groups.UpsertNode(ctx, proto.Node{Address: config.Address, Capacity: 0, State: proto.NodeStateError})
					xl.Errorf("NewBaseGroups s.initGroups error: %s", err)
					return nil, errors.Wrap(err, "initGroups")
				}
				xl.Infof("load data %v , use time %v", cnt, time.Since(start))
				break
			}
			if node.State != proto.NodeStateReady {
				allInitialized = false
				break
			}
		}
		xl.Debugf("Nodes: %#v", s.Nodes)
		if allInitialized {
			return s, nil
		}
		time.Sleep(time.Duration(defaultInitInterval) * time.Second)
		intiWait += defaultInitInterval
	}
	s.Groups.UpsertNode(ctx, proto.Node{Address: config.Address, Capacity: 0, State: proto.NodeStateError})
	return nil, errors.New("init cluster failed")
}

func (s *BaseGroups) initGroups(ctx context.Context) (cnt int, err error) {
	xl := xlog.FromContextSafe(ctx)
	names, err := s.Groups.All(ctx)
	if err != nil {
		xl.Errorf("initGroups: database.AllGroups error: %s", err)
		return 0, err
	}

	var (
		node = proto.Node{
			Address:  s.GroupsConfig.Address,
			Capacity: proto.NodeCapacity(s.GroupsConfig.Sets.BlockSize * s.GroupsConfig.Sets.BlockNum),
			State:    proto.NodeStateInitializing,
		}
		rg proto.HashKeyRange
	)

	if s.GroupsConfig.ClusterMode {
		if err = s.Groups.UpsertNode(ctx, node); err != nil {
			xl.Errorf("Groups.UpsertNode error: %s", err)
			return 0, errors.Wrap(err, "Groups.UpsertNode")
		}
		if s.Nodes, err = s.Groups.AllNodes(ctx); err != nil {
			xl.Errorf("Groups.AllNodes error: %s", err)
			return 0, errors.Wrap(err, "Groups.AllNodes")
		}
		if len(s.Nodes) != s.GroupsConfig.ClusterSize {
			xl.Error("initGroups dismatch cluster_size, quit to wait for other cluster node up...")
			return 0, errors.New("cluster size mismatch")
		}
		for index, n := range s.Nodes {
			if node.Address == n.Address {
				length := (1<<32 + len(s.Nodes) - 1) / len(s.Nodes)
				rg[0] = proto.FeatureHashKey(uint32(index) * uint32(length))
				if index == len(s.Nodes)-1 {
					rg[1] = 1<<32 - 1
				} else {
					rg[1] = proto.FeatureHashKey(uint32(index+1) * uint32(length))
				}
			}
		}
	}

	for _, name := range names {
		group, err := s.Groups.Get(ctx, name)
		if err != nil {
			xl.Errorf("Groups.Get error: %s", err)
			return 0, err
		}
		gConfig := group.Config(ctx)

		// cluster node 0  must check if all the features with hash_key
		if s.GroupsConfig.ClusterMode && s.GroupsConfig.Address == s.Nodes[0].Address {
			var withoutKey int
			if withoutKey, err = group.CountWithoutHashKey(ctx); err != nil {
				xl.Errorf("Groups.CountWithoutHashKey error: %s", err)
				return 0, err
			}
			if withoutKey > 0 {
				if err = group.EnsureHashKey(ctx, func(id proto.FeatureID) proto.FeatureHashKey {
					return proto.FeatureHashKey(crc32.ChecksumIEEE([]byte(id)))
				}); err != nil {
					xl.Errorf("BaseGroups.EnsureHashKey error: %s", err)
					return 0, err
				}
				// sleep defaultInitInterval*2, ensure other nodes catch NodeEnsureHashKey state
				node.State = proto.NodeEnsureHashKey
				if err = s.Groups.UpsertNode(ctx, node); err != nil {
					xl.Errorf("Groups.UpsertNode error: %s", err)
					return 0, errors.Wrap(err, "Groups.UpsertNode")
				}
				time.Sleep(time.Duration(defaultInitInterval*2) * time.Second)
			}
		}

		n, err := group.Count(ctx, rg)
		if err != nil {
			xl.Errorf("Groups.Count error: %s", err)
			return 0, err
		}
		cnt += n
		node.Capacity -= proto.NodeCapacity(n)
		if err = s.Groups.UpsertNode(ctx, node); err != nil {
			xl.Errorf("Groups.UpsertNode error: %s", err)
			return 0, errors.Wrap(err, "Groups.UpsertNode")
		}

		set, err := s.Sets.Get(ctx, search.SetName(name))
		if err != nil {
			if err == search.ErrFeatureSetNotFound {
				setConfig := s.genSetsConfig(gConfig)
				setConfig.Capacity = n
				err = s.Sets.New(ctx, search.SetName(name), setConfig, search.SetState(proto.GroupCreated))
				if err != nil {
					xl.Errorf("fail to create set %s, err: %v", name, err)
					return 0, err
				}
				set, err = s.Sets.Get(ctx, search.SetName(name))
				if err != nil {
					xl.Errorf("fail to get set %s, err: %v", name, err)
					return 0, err
				}
				if err = group.Iter(ctx, rg, set.Add); err != nil {
					xl.Errorf("fail to iter add features to group %s, err: %s", name, err)
					return 0, err
				}
				if err = set.SetState(ctx, search.SetState(proto.GroupInitialized)); err != nil {
					xl.Errorf("fail to update set %s to state GroupInitialized, err: %s", name, err)
				}
				continue
			}
			xl.Errorf("Sets.Get error: %s", err)
			return 0, err
		}

		sConfig := set.Config(ctx)

		if gConfig.Version == sConfig.Version {
			continue
		}

		if sConfig.Version != 0 {
			if err = set.Destroy(ctx); err != nil {
				xl.Errorf("fail to destroy set %s, err: %v", name, err)
				return 0, err
			}
			setConfig := s.genSetsConfig(gConfig)
			setConfig.Capacity = n

			err = s.Sets.New(ctx, search.SetName(name), setConfig, search.SetState(proto.GroupCreated))
			if err != nil {
				xl.Errorf("fail to create set %s, err: %v", name, err)
				return 0, err
			}
			set, err = s.Sets.Get(ctx, search.SetName(name))
			if err != nil {
				xl.Errorf("fail to get set %s, err: %v", name, err)
				return 0, err
			}
			if err = group.Iter(ctx, rg, group.Add); err != nil {
				xl.Errorf("fail to iter add features to group %s, err: %s", name, err)
				return 0, err
			}
			if err = set.SetState(ctx, search.SetState(proto.GroupInitialized)); err != nil {
				xl.Errorf("fail to update set %s to state GroupInitialized, err: %s", name, err)
			}
		}
	}
	node.State = proto.NodeStateReady
	err = s.Groups.UpsertNode(ctx, node)
	return cnt, err
}

func (s *BaseGroups) genSetsConfig(groupConfig proto.GroupConfig) search.Config {
	setConfig := search.Config{
		Precision: groupConfig.Precision,
		Dimension: groupConfig.Dimension,
		Capacity:  groupConfig.Capacity,
	}
	mergo.Merge(&setConfig, s.GroupsConfig.Sets)
	return setConfig
}

func (s *BaseGroups) New(ctx context.Context, internal bool, name proto.GroupName, config proto.GroupConfig) (err error) {
	xl := xlog.FromContextSafe(ctx)
	cfg := config
	if len(name) == 0 {
		err = errors.New("Invalid Group Name")
		return
	}
	if !internal {
		_, err = s.Groups.Get(ctx, name)
		if err != manager.ErrGroupNotExist {
			return
		}
		if s.GroupsConfig.ClusterMode {
			cfg.Capacity = cfg.Capacity / len(s.Nodes)
			for _, node := range s.Nodes {
				if node.Address != s.GroupsConfig.Address {
					if err = s.baseFeature.CreateGroup(ctx, node.Address, name, cfg); err != nil {
						return
					}
				}
			}
		}
	}
	setConfig := s.genSetsConfig(cfg)
	err = s.Sets.New(ctx, search.SetName(name), setConfig, search.SetState(proto.GroupCreated))
	if err != nil {
		xl.Errorf("Set New error: %s", err)
		return
	}
	if !internal {
		err = s.Groups.New(ctx, name, config)
		if err != nil {
			set, sErr := s.Sets.Get(ctx, search.SetName(name))
			if sErr == nil {
				sErr = set.Destroy(ctx)
			}
			if sErr != nil {
				xl.Errorf("Fatal error: fail to rollback deleting set %s", sErr)
			}
			xl.Errorf("Group New error: %s", err)
			return
		}
	}
	return
}

func (s *BaseGroups) Get(ctx context.Context, name proto.GroupName) (feature_group.IGroup, error) {
	if len(name) == 0 {
		return nil, errors.New("Invalid Group Name")
	}
	group, err := s.Groups.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	set, err := s.Sets.Get(ctx, search.SetName(name))
	if err != nil {
		return nil, err
	}
	gs := &_BaseGroup{
		Name:        name,
		group:       group,
		set:         set,
		manager:     s,
		nodes:       s.Nodes,
		baseFeature: s.baseFeature,
	}
	return gs, nil
}

func (s *BaseGroups) All(ctx context.Context) ([]proto.GroupName, error) {
	xl := xlog.FromContextSafe(ctx)
	names, err := s.Groups.All(ctx)
	if err != nil {
		xl.Errorf("_BaseGroup.All s.Groups.All error: %s", err)
		return nil, err
	}
	return names, nil
}

//------------------- _BaseGroup -------------------//
var _ feature_group.IGroup = new(_BaseGroup)

type _BaseGroup struct {
	Name        proto.GroupName
	group       manager.Group
	set         search.Set
	manager     *BaseGroups
	nodes       []proto.Node
	baseFeature feature.BaseFeature
}

func checkNodes(nodes []proto.Node, key proto.FeatureHashKey) proto.NodeAddress {
	length := uint32((1<<32 + len(nodes) - 1) / len(nodes))
	return nodes[uint32(key)/length].Address
}

func (s *_BaseGroup) Destroy(ctx context.Context, internal bool) error {
	var (
		xl         = xlog.FromContextSafe(ctx)
		gErr, sErr error
	)
	if !internal {
		if s.manager.GroupsConfig.ClusterMode {
			for _, node := range s.nodes {
				if node.Address != s.manager.GroupsConfig.Address {
					if sErr = s.baseFeature.RemoveGroup(ctx, node.Address, s.Name); sErr != nil {
						return sErr
					}
				}
			}
		}
		gErr = s.group.Destroy(ctx)
	}
	sErr = s.set.Destroy(ctx)
	if !internal {
		if gErr != nil {
			xl.Errorf("_BaseGroup.Destroy group.Destroy error: %s", gErr)
			return gErr
		}
	}
	if sErr != nil {
		xl.Errorf("_BaseGroup.Destroy set.Destroy error: %s", sErr)
		return sErr
	}
	return nil
}

func (s *_BaseGroup) Count(ctx context.Context) (count int, err error) {
	xl := xlog.FromContextSafe(ctx)
	count, err = s.group.Count(ctx, proto.HashKeyRange{})
	if err != nil {
		xl.Errorf("_BaseGroup.Count group.Count error: %s", err)
		return
	}
	return
}

func (s *_BaseGroup) CountTags(ctx context.Context) (count int, err error) {
	xl := xlog.FromContextSafe(ctx)
	count, err = s.group.CountTags(ctx)
	if err != nil {
		xl.Errorf("_BaseGroup.CountTags group.CountTags error: %s", err)
		return
	}
	return
}

func (s *_BaseGroup) Config(ctx context.Context) proto.GroupConfig {
	return s.group.Config(ctx)
}

func (s *_BaseGroup) Add(ctx context.Context, internal bool, features ...proto.Feature) (err error) {
	xl := xlog.FromContextSafe(ctx)
	featureIDs := make([]proto.FeatureID, 0, len(features))
	needRollback := true
	for _, feature := range features {
		if feature.ID == "" {
			xl.Errorf("_BaseGroup.Add feature.ID empty")
			return errors.New("feature.ID empty")
		}
		featureIDs = append(featureIDs, feature.ID)
	}
	defer func() {
		if err != nil && needRollback {
			// 如果是 单机模式 或者 多机模式下的主节点, 则删除全部feature
			if !internal {
				xl.Debugf("_BaseGroup.Add rollback db")
				e := s.group.Delete(ctx, featureIDs...)
				if e != nil {
					xl.Errorf("_BaseGroup.Add rollback Group.Delete error: %s", e.Error())
				}
			}
			// 错误则删除全部的features,
			// 单机模式 多机模式的从节点, 没有冗余删除, 主节点有冗余删除, 不过是幂等的
			_, e := s.set.Delete(ctx, featureIDs...)
			if e != nil {
				xl.Errorf("_BaseGroup.Add rollback Set.Delete error: %s", e.Error())
			}
		}
	}()
	if !internal {
		err = s.group.Add(ctx, features...)
		if err != nil {
			needRollback = false
			xl.Errorf("_BaseGroup.Add Group.Add error: %s", err)
			return
		}
		if s.manager.GroupsConfig.ClusterMode {
			reqs := make(map[proto.NodeAddress][]proto.Feature, 0)
			for _, feature := range features {
				address := checkNodes(s.nodes, feature.HashKey)
				if _, exist := reqs[address]; !exist {
					reqs[address] = make([]proto.Feature, 0)
				}
				reqs[address] = append(reqs[address], feature)
			}
			defer func() {
				// 多机模式主节点广播所有删除的请求, 除了自己. 且就算没有用成功插入的, 接受了删除的请求也是幂等的
				if err != nil {
					for address, req := range reqs {
						if address != s.manager.GroupsConfig.Address {
							featureIDs := make([]proto.FeatureID, 0, len(req))
							for _, feature := range req {
								featureIDs = append(featureIDs, feature.ID)
							}
							if e := s.baseFeature.DeleteFeature(ctx, address, s.Name, featureIDs...); e != nil {
								xl.Errorf("_BaseGroup.Add rollback baseFeature.DeleteFeature node %s, group: %s,  error: %s", address, s.Name, e.Error())
							}
						}
					}
				}
			}()
			for address, req := range reqs {
				if address != s.manager.GroupsConfig.Address {
					if err = s.baseFeature.AddFeature(ctx, address, s.Name, req...); err != nil {
						xl.Errorf("_BaseGroup.Add baseFeature.AddFeature to node %s, group: %s,  error: %s", address, s.Name, err)
						return
					}
				} else {
					if !s.set.SpaceAvailable(ctx, len(req)) {
						err = search.ErrBlockIsFull
						xl.Errorf("_BaseGroup.Add Set.Add error: %s", err)
						return
					}
					err = s.set.Add(ctx, req...)
					if err != nil {
						xl.Errorf("_BaseGroup.Add Set.Add error: %s", err)
						return
					}
				}
			}
			return
		}
	}

	if !s.set.SpaceAvailable(ctx, len(features)) {
		err = search.ErrBlockIsFull
		xl.Errorf("_BaseGroup.Add Set.Add error: %s", err)
		return
	}
	err = s.set.Add(ctx, features...)
	if err != nil {
		xl.Errorf("_BaseGroup.Add Set.Add error: %s", err)
		return
	}
	return
}

func (s *_BaseGroup) Delete(ctx context.Context, internal bool, ids ...proto.FeatureID) (deleted []proto.FeatureID, err error) {
	xl := xlog.FromContextSafe(ctx)
	if !internal {
		deleted, err = s.group.Exist(ctx, ids...)
		if err != nil {
			xl.Errorf("_BaseGroup.Delete Group.Exist error: %s", err)
			return
		}
		err = s.group.Delete(ctx, ids...)
		if err != nil {
			xl.Errorf("_BaseGroup.Delete Group.Delete error: %s", err)
			return
		}
		if s.manager.GroupsConfig.ClusterMode {
			reqs := make(map[proto.NodeAddress][]proto.FeatureID, 0)
			for _, id := range ids {
				key := proto.FeatureHashKey(crc32.ChecksumIEEE([]byte(id)))
				address := checkNodes(s.nodes, key)
				if _, exist := reqs[address]; !exist {
					reqs[address] = make([]proto.FeatureID, 0)
				}
				reqs[address] = append(reqs[address], id)
			}
			for address, req := range reqs {
				if address != s.manager.GroupsConfig.Address {
					if err = s.baseFeature.DeleteFeature(ctx, address, s.Name, req...); err != nil {
						xl.Errorf("_BaseGroup.Delete baseFeature.Delete to node %s, group: %s,  error: %s", address, s.Name, err)
						return nil, err
					}
				} else {
					_, err = s.set.Delete(ctx, req...)
					if err != nil {
						xl.Errorf("_BaseGroup.Delete Set.Delete error: %s", err)
						return
					}
				}
			}
			return
		}
	}
	_, err = s.set.Delete(ctx, ids...)
	if err != nil {
		xl.Errorf("_BaseGroup.Delete Set.Delete error: %s", err)
		return
	}
	return
}

func (s *_BaseGroup) Update(ctx context.Context, internal bool, features ...proto.Feature) (err error) {
	xl := xlog.FromContextSafe(ctx)
	for _, feature := range features {
		if feature.ID == "" {
			xl.Errorf("_BaseGroup.Add feature.ID empty")
			return errors.New("feature.ID empty")
		}
	}
	if !internal {
		err = s.group.Update(ctx, features...)
		if err != nil {
			xl.Errorf("_BaseGroup.Update Group.Update error: %s", err)
			return
		}
		if s.manager.GroupsConfig.ClusterMode {
			reqs := make(map[proto.NodeAddress][]proto.Feature, 0)
			for _, feature := range features {
				address := checkNodes(s.nodes, feature.HashKey)
				if _, exist := reqs[address]; !exist {
					reqs[address] = make([]proto.Feature, 0)
				}
				reqs[address] = append(reqs[address], feature)
			}
			for address, req := range reqs {
				if address != s.manager.GroupsConfig.Address {
					if err = s.baseFeature.UpdateFeature(ctx, address, s.Name, req...); err != nil {
						xl.Errorf("_BaseGroup.Update baseFeature.UpdateFeature to node %s, group: %s,  error: %s", address, s.Name, err)
						return err
					}
				} else {
					err = s.set.Update(ctx, req...)
					if err != nil {
						xl.Errorf("_BaseGroup.Update Set.Update error: %s", err)
						return
					}
				}
			}
			return
		}
	}
	err = s.set.Update(ctx, features...)
	if err != nil {
		xl.Errorf("_BaseGroup.Update Set.Update error: %s", err)
		return
	}
	return
}

func (s *_BaseGroup) Tags(ctx context.Context, marker string, limit int) ([]proto.GroupTagInfo, string, error) {
	xl := xlog.FromContextSafe(ctx)
	if marker != "" {
		var err error
		marker, err = decodeMarker(marker)
		if err != nil {
			return nil, "", errors.New("Invalid marker")
		}
	}
	tags, nextMarker, err := s.group.Tags(ctx, marker, limit)
	if err != nil {
		xl.Errorf("_BaseGroup.Tags group.Tags error: %s", err)
		return nil, "", err
	}
	return tags, encodeMarker(nextMarker), nil
}

func (s *_BaseGroup) FilterByTag(ctx context.Context, tag proto.FeatureTag, marker string, limit int) ([]proto.Feature, string, error) {
	xl := xlog.FromContextSafe(ctx)
	features, nextMarker, err := s.group.FilterByTag(ctx, tag, marker, limit)
	if err != nil {
		xl.Errorf("_BaseGroup.FilterByTag group.FilterByTag error: %s", err)
		return nil, "", err
	}
	return features, nextMarker, nil
}

func (s *_BaseGroup) Search(ctx context.Context, internal bool,
	threshold float32, limit int,
	features ...proto.FeatureValue,
) (
	ret [][]feature_group.FeatureSearchRawRespItem,
	err error,
) {
	var (
		xl                    = xlog.FromContextSafe(ctx)
		wg                    = sync.WaitGroup{}
		searchResult, results [][]feature_group.FeatureSearchItem
		mutex                 sync.Mutex
		err1                  error
	)

	if len(features) == 0 {
		return
	}
	if !internal && s.manager.GroupsConfig.ClusterMode {
		results = make([][]feature_group.FeatureSearchItem, len(features))
		for index, node := range s.nodes {
			if node.Address != s.manager.GroupsConfig.Address {
				wg.Add(1)
				go func(ctx context.Context, index int, address proto.NodeAddress) {
					defer wg.Done()
					result, e := s.baseFeature.SearchFeature(ctx, address, s.Name, threshold, limit, features...)
					if e != nil {
						xlog.FromContextSafe(ctx).Errorf("_BaseGroup.Search baseFeature.SearchFeature to node %s, group %s, error: %s", address, s.Name, e.Error())
						err1 = e
					}
					mutex.Lock()
					defer mutex.Unlock()
					for i, r := range result {
						results[i] = append(result[i], r...)
					}
					return
				}(util.SpawnContext(ctx), index, node.Address)
			}
		}
	}

	searchResult, err = s.set.Search(ctx, threshold, limit, features...)
	if err != nil {
		xl.Errorf("_BaseGroup.Search set.Search error: %s", err)
		return nil, err
	}
	if !internal && s.manager.GroupsConfig.ClusterMode {
		wg.Wait()
		if err1 != nil {
			return nil, err1
		}
		for i, r := range searchResult {
			results[i] = append(results[i], r...)
			_, searchResult[i] = search.MaxNFeatureSearchResult(results[i], limit)
		}
	}
	for r, row := range searchResult {
		ret = append(ret, make([]feature_group.FeatureSearchRawRespItem, 0))
		for _, item := range row {
			if len(item.ID) == 0 {
				continue
			}
			f, err := s.group.Get(ctx, item.ID)
			if err != nil {
				xl.Errorf("_BaseGroup.Search manager.Get id(%s) error: %s", item.ID, err)
				return nil, err
			}

			score := item.Score
			if score < 0 {
				score = 0
			}
			if s.manager.GroupsConfig.SigmoidConfig != nil && s.manager.GroupsConfig.SigmoidConfig.ThresholdNew != 0 {
				score = float32(sigmoid.Sigmoid(
					float64(score),
					float64(threshold),
					s.manager.GroupsConfig.SigmoidConfig.ThresholdNew,
					s.manager.GroupsConfig.SigmoidConfig.SigmoidA))
			}
			ret[r] = append(ret[r], feature_group.FeatureSearchRawRespItem{
				Value: f,
				Score: score,
			})
		}
	}
	return
}

func (s *_BaseGroup) Compare(ctx context.Context,
	target feature_group.IGroup,
	threshold float32, limit int,
) (
	ret []feature_group.BaseCompareResult,
	err error,
) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	targetGroup, ok := target.(*_BaseGroup)
	if !ok {
		return nil, errors.New("target is invalid base group")
	}

	results, err := s.set.Compare(ctx, threshold, limit, targetGroup.set)
	if err != nil {
		xl.Errorf("_BaseGroup.Compareset.Compare error: %s", err.Error())
		return nil, err
	}
	for _, result := range results {
		if len(result.Faces) == 0 {
			continue
		}
		f, err := targetGroup.group.Get(ctx, result.ID)
		if err != nil {
			xl.Errorf("_BaseGroup.Compare target group %s, manager.group.Get id(%s) error: %s", targetGroup.Name, result.ID, err)
			return nil, err
		}
		r := feature_group.BaseCompareResult{
			ID:          f.ID,
			Tag:         f.Tag,
			Desc:        f.Desc,
			BoundingBox: f.BoundingBox,
		}
		for _, face := range result.Faces {
			ff, err := s.group.Get(ctx, face.ID)
			if err != nil {
				xl.Errorf("_BaseGroup.Search manager.Get id(%s) error: %s", face.ID, err)
				return nil, err
			}
			score := face.Score
			if score < 0 {
				score = 0
			}
			r.Faces = append(r.Faces, feature_group.FeatureCompareRespItem{
				ID:          ff.ID,
				Tag:         ff.Tag,
				Desc:        ff.Desc,
				BoundingBox: ff.BoundingBox,
				Score:       score,
			})
		}
		ret = append(ret, r)
	}

	return
}
