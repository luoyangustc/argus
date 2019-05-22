:: 设置环境变量
 
:: 关闭终端回显
@echo off
 set ORG_GOPATH=%GOPATH%
@echo ====current environment:
@echo %ORG_GOPATH%
 
:: 添加环境变量,即在原来的环境变量后加上英文状态下的分号和路径
set MY_PATH=D:\Workspace\AIProjects\nvs\platform\projects\api_serv
set GOPATH=%ORG_GOPATH%;%MY_PATH%
@echo ====new environment:
@echo %GOPATH%

set GOARCH=%GOHOSTARCH%
set GOOS=%GOHOSTOS%
 
cd D:\Workspace\AIProjects\nvs\platform\projects\api_serv\src

go clean -i ./...

go build -v httpserv
go install httpserv

go build -v api_serv
go install api_serv

set GOPATH=%ORG_GOPATH%
@echo %GOPATH%

cd %MY_PATH%
 
pause
