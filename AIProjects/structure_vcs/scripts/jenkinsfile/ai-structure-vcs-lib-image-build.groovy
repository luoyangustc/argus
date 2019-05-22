
pipeline {
    agent {
        node('CS47')
    }
    environment {
        BUILD_DATE = sh(script: "date +'%Y%m%d%H%M%S' | tr -dc '[[:print:]]'", returnStdout: true)
        IMAGE_NAME = "ai_strcuture_vcs"
        IMAGE_TAG = "${BUILD_DATE}-v${BUILD_ID}"
    }
    stages {
        stage('clean workspace') {
            steps {
                cleanWs()
            }
        }
        stage('git pull') {
            steps {
                dir('AIProjects') {
                    checkout([$class: 'GitSCM', 
                    branches: [[name: "${BRANCH}"]],
                    userRemoteConfigs: [[
                        credentialsId: '71382492-af44-4193-8192-eb846fd45f86', 
                        refspec: '+refs/tags/*:refs/remotes/origin/tags/*', 
                        url: "git@github.com:${UPSTREAM}/AIProjects.git"]],
                        browser: [$class: 'GithubWeb', repoUrl: "https://github.com/${UPSTREAM}/AIProjects/tree/${BRANCH}"]
                    ])
                }
            }
        }
        stage('build & push image') {
            environment {
                AVAPRD_USERNAME = "avaprd@qiniu.com"
                AVAPRD_PASSWORD = credentials('AVAPRD_PASSWORD')
            }
            steps {
                sh('''
                set -ex
                docker login reg.qiniu.com -u ${AVAPRD_USERNAME} -p ${AVAPRD_PASSWORD}
                cd ${WORKSPACE}/AIProjects/structure_vcs
                docker build . -f docker/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG}
                docker tag ${IMAGE_NAME}:${IMAGE_TAG} reg.qiniu.com/avaprd/${IMAGE_NAME}:${IMAGE_TAG}
                docker push reg.qiniu.com/avaprd/${IMAGE_NAME}:${IMAGE_TAG}
                docker rmi reg.qiniu.com/avaprd/${IMAGE_NAME}:${IMAGE_TAG}
                docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
                ''')
            }
        }
    }
}