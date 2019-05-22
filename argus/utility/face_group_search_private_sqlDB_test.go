package utility

import (
	"context"
	"database/sql"
	"os"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/stretchr/testify/assert"
)

func TestPFaceGroupInDB(t *testing.T) {

	db, err := sql.Open("sqlite3", "./foo.db")
	assert.NoError(t, err)
	defer os.Remove("./foo.db")
	defer db.Close()

	NewPFaceGroupManagerInDB(db)

	var (
		uid uint32 = 1111
		gid        = "foo1"
	)
	g := pfaceGroupInDB{db: db, uid: uid, id: gid}
	db.Exec("DELETE FROM faces WHERE uid = $1 AND gid = $2", uid, gid)

	var fid string
	{
		ctx := context.Background()
		_, err := g.Add(ctx, []_FaceItem{
			_FaceItem{
				Name:    "abc",
				Feature: []byte("abcdefg"),
			},
		})
		assert.NoError(t, err)
		items, err := g.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(items))
		assert.Equal(t, "abc", items[0].Name)
		assert.Equal(t, "abcdefg", string(items[0].Feature))
		fid = items[0].ID
	}
	{
		ctx := context.Background()
		assert.NoError(t, g.Del(ctx, []string{fid}))
		items, err := g.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(items))
	}
}

func TestPFaceGroupManagerInDB(t *testing.T) {

	db, err := sql.Open("sqlite3", "./foo.db")
	assert.NoError(t, err)
	defer os.Remove("./foo.db")
	defer db.Close()

	m, _ := NewPFaceGroupManagerInDB(db)

	var (
		uid uint32 = 1121
		gid        = "foo1"
	)
	db.Exec("DELETE FROM groups WHERE uid = $1 AND id = $2", uid, gid)
	db.Exec("DELETE FROM faces WHERE uid = $1 AND gid = $2", uid, gid)

	{
		ctx := context.Background()
		g, err := m.New(ctx, uid, gid)
		assert.NoError(t, err)
		items, err := g.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(items))
	}
	{
		ctx := context.Background()
		g, err := m.Get(ctx, uid, gid)
		assert.NoError(t, err)
		items, err := g.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(items))
	}
	{
		ctx := context.Background()
		assert.NoError(t, m.Remove(ctx, uid, gid))
		_, err := m.Get(ctx, uid, gid)
		assert.EqualError(t, err, sql.ErrNoRows.Error())
	}
}
