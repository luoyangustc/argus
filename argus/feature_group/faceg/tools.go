package faceg

// OLD:
//
// fdbgroups:
// bson.M{"uid": uid, "id": id}
//
// fdbfaces:
// Uid     uint32 `bson:"uid"`
// Gid     string `bson:"gid"`
// ID      string `bson:"id"`
// Name    string `bson:"name"`
// Feature []byte `bson:"feature"`
// Static  bool   `bson:"static"`
//
// GROUP:
// fg_groups:
// bson.M{"uid": uid, "id": id, "hid": hubID}
// fg_faces:
// UID    uint32 `bson:"uid"`
// Gid    string `bson:"gid"`
// ID     string `bson:"id"`
// Name   string `bson:"name"`
// Backup string `bson:"backup"`
//
// HUB:
// fg_hub_hubs:
// 	ID        bson.ObjectId `bson:"_id,omitempty"`
// ChunkSize uint64        `bson:"chunk_size"`
// Cursor    int           `bson:"cursor"`
// Version   HubVersion    `bson:"version"`
// fg_hub_features:
// Hid     HubID      `bson:"hid"`
// Fid     FeatureID  `bson:"fid"`
// Index   int        `bson:"index"`
// Version HubVersion `bson:"version"`
// Feature []byte     `bson:"feature"`

// func CopyFromOldMgo(ctx context.Context,
// 	srcConf, groupConf, hubConf *mgoutil.Config,
// ) error {

// 	var (
// 		xl  = xlog.FromContextSafe(ctx)
// 		src = struct {
// 			Groups mgoutil.Collection `coll:"fdbgroups"`
// 			Faces  mgoutil.Collection `coll:"fdbfaces"`
// 		}{}

// 		group _FaceGroupManager
// 	)

// 	{
// 		sess, err := mgoutil.Open(&src, srcConf)
// 		if err != nil {
// 			return err
// 		}
// 		defer sess.Close()
// 		sess.SetPoolLimit(10)
// 	}
// 	{
// 		hub, _ := FG.NewHubInMgo(hubConf,
// 			&struct {
// 				Hubs     mgoutil.Collection `coll:"fg_hub_hubs"`
// 				Features mgoutil.Collection `coll:"fg_hub_features"`
// 			}{},
// 		)
// 		group, _ = NewFaceGroupManagerInMgo(groupConf, hub)
// 	}

// 	{
// 		iter := src.Groups.Find(bson.M{}).Iter()
// 		for {
// 			var e = struct {
// 				UID uint32 `bson:"uid"`
// 				ID  string `bson:"id"`
// 			}{}
// 			if ok := iter.Next(&e); !ok {
// 				break
// 			}

// 			_, _ = group.New(ctx, e.UID, e.ID, FG.EmptyFeatureVersion)
// 			xl.Infof("GROUP: %d %s", e.UID, e.ID)
// 		}
// 	}
// 	{
// 		iter := src.Faces.Find(bson.M{}).Sort("uid", "gid").Iter()
// 		for {
// 			var e = struct {
// 				Uid     uint32 `bson:"uid"`
// 				Gid     string `bson:"gid"`
// 				ID      string `bson:"id"`
// 				Name    string `bson:"name"`
// 				Feature []byte `bson:"feature"`
// 				Static  bool   `bson:"static"`
// 			}{}
// 			if ok := iter.Next(&e); !ok {
// 				break
// 			}

// 			fg, _ := group.Get(ctx, e.Uid, e.Gid)
// 			fg.Add(ctx,
// 				[]_FaceItem{{ID: e.ID, Name: e.Name}},
// 				[][]byte{e.Feature},
// 			)
// 			xl.Infof("FACE: %d %s %s", e.Uid, e.Gid, e.ID)
// 		}
// 	}

// 	return nil
// }
