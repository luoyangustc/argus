package proto

// NodeAddress: <ip:port>
type NodeAddress string
type NodeCapacity uint64
type NodeState uint32

const (
	NodeStateInitializing NodeState = iota
	NodeStateReady
	NodeStateError
	NodeEnsureHashKey
)

type Node struct {
	Address  NodeAddress  `bson:"address"`
	Capacity NodeCapacity `bson:"capacity"`
	State    NodeState    `bson:"state"`
}
