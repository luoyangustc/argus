package util

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path"
	"strings"
)

// copyFile copies the contents of the file named src to the file named
// by dst. The file will be created if it does not already exist. If the
// destination file exists, all it's contents will be replaced by the contents
// of the source file.
func CopyFile(src, dst string) (err error) {
	in, err := os.Open(src)
	if err != nil {
		return
	}
	defer func() {
		closeErr := in.Close()
		if err == nil {
			err = closeErr
		}
	}()

	srcInfo, err := in.Stat()
	if err != nil {
		return
	}
	mode := srcInfo.Mode()

	out, err := os.Create(dst)
	if err != nil {
		return
	}

	defer func() {
		closeErr := out.Close()
		if err == nil {
			err = closeErr
		}
	}()

	_, err = io.Copy(out, in)
	if err != nil {
		return
	}

	err = out.Sync()
	if err != nil {
		return
	}

	err = out.Chmod(mode)
	return
}

func ExtractTar(workspece string, tarFile string, gzipped bool) (files map[string]string, err error) {
	srcFile, err := os.Open(tarFile)
	if err != nil {
		return nil, err
	}
	defer srcFile.Close()

	var src io.Reader = srcFile
	if gzipped {
		gz, e := gzip.NewReader(srcFile)
		if e != nil {
			err = e
			return
		}
		defer gz.Close()
		src = gz
	}

	tr := tar.NewReader(src)
	files = make(map[string]string, 0)
	for {
		hdr, err := tr.Next()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, err
			}
		}

		if strings.HasPrefix(hdr.Name, "./.") {
			//mac下面的tar的一个特点，每个压入的文件会有一个对应的隐藏文件，需要过滤这种文件以免引起错误
			//https://stackoverflow.com/questions/8766730/tar-command-in-mac-os-x-adding-hidden-files-why
			continue
		}
		filename := path.Join(workspece, hdr.Name)
		if hdr.Typeflag == byte(tar.TypeDir) {
			if _, err := os.Stat(filename); err != nil {
				if err := os.MkdirAll(filename, 0776); err != nil {
					return nil, err
				}
			}
			continue
		}
		stat, err := os.Stat(filename)
		if err == nil && stat.Size() == hdr.Size {
			files[hdr.Name] = filename
			continue
		}
		err = func() error {
			file, err := os.Create(filename)
			if err != nil {
				return err
			}
			defer file.Close()
			_, err = io.Copy(file, tr)
			if err != nil {
				return err
			}
			return nil
		}()
		if err != nil {
			return nil, err
		}
		files[hdr.Name] = filename

	}
	return
}
