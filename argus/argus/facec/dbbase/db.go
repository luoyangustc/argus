package dbbase

import (
	"errors"

	"gopkg.in/mgo.v2"

	"github.com/qiniu/xlog.v1"
)

// MaxLimit define the max count returned for the query
const MaxLimit = 500

type DB struct {
	Address  string `json:"address"`
	Database string `json:"database"`
}

var dbConfig *DB

// Conn the db connection
type Conn struct {
	session *mgo.Session
	db      *mgo.Database
}

// C return the collection by name
func (conn *Conn) C(name string) *mgo.Collection {
	return conn.db.C(name)
}

// Close the connection
func (conn *Conn) Close() {
	conn.session.Close()
}

// Init do the initialization of db
func Init(config *DB) error {
	if config == nil {
		return errors.New("empty config")
	}
	dbConfig = config

	_, err := mgo.Dial(dbConfig.Address)
	if err != nil {
		return err
	}
	return nil
}

// NewConn create new connection
func NewConn() *Conn {
	session, err := mgo.Dial(dbConfig.Address)
	if err != nil {
		xlog.Errorf("", "new connection error", err)
		return nil
	}

	db := session.DB(dbConfig.Database)

	return &Conn{session, db}
}
