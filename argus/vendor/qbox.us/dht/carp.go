package dht

import (
	"crypto/sha1"
	"encoding/hex"
	"sort"
)

type Carp struct {
	nodes NodeInfos
}

func NewCarp(nodes NodeInfos) Interface {
	crap := &Carp{}
	crap.Setup(nodes)
	return crap
}

func (p *Carp) Setup(nodes NodeInfos) {
	p.nodes = nodes
}

func (p *Carp) Nodes() NodeInfos {
	return p.nodes
}

// len(routes) <= ttl
func (p *Carp) Route(key []byte, ttl int) (routers RouterInfos) {
	n := ttl
	if n > len(p.nodes) {
		n = len(p.nodes)
	}
	nodes := make([]carpNode, len(p.nodes))
	for i, node := range p.nodes {
		hash := sha1.New()
		hash.Write(key)
		hash.Write(node.Key)
		nodes[i] = carpNode{node.Host, hex.EncodeToString(hash.Sum(nil))}
	}
	sort.Sort(carpSlice(nodes))
	rs := make([]RouterInfo, n)
	for i := 0; i < n; i++ {
		rs[i].Host = nodes[i].host
		rs[i].Metrics = i + 1
	}
	return rs
}

type carpNode struct {
	host string
	hash string
}
type carpSlice []carpNode

func (p carpSlice) Len() int           { return len(p) }
func (p carpSlice) Less(i, j int) bool { return p[i].hash < p[j].hash }
func (p carpSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
