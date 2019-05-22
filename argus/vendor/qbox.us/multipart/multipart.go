package multipart

import (
	"io"
	"log"
	"mime/multipart"
	"os"
)

func Write(w *multipart.Writer, params map[string][]string) error {

	for key, param1 := range params {
		param := param1[0]
		if len(param) > 0 && param[0] == '@' {
			file := param[1:]
			fw, err := w.CreateFormFile(key, file)
			if err != nil {
				log.Println("CreateFormFile failed:", err)
				return err
			}
			fd, err := os.Open(file)
			if err != nil {
				log.Println("Open file failed:", err)
				return err
			} else {
				_, err = io.Copy(fw, fd)
				fd.Close()
				if err != nil {
					log.Println("Copy file failed:", err)
					return err
				}
			}
		} else {
			err := w.WriteField(key, param)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func Open(params map[string][]string) (r io.ReadCloser, conentType string, err error) {

	r, w1 := io.Pipe()
	w := multipart.NewWriter(w1)
	conentType = w.FormDataContentType()

	go func() {
		err := Write(w, params)
		w.Close()
		w1.CloseWithError(err)
	}()

	return
}
