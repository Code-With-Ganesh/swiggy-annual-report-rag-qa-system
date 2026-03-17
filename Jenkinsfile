pipeline {
  agent any

  options {
    ansiColor('xterm')
    timestamps()
  }

  parameters {
    string(name: 'ADMIN_CIDR', defaultValue: '203.0.113.10/32', description: 'Trusted CIDR for SSH in secure Terraform stack')
    booleanParam(name: 'RUN_TERRAFORM_PLAN', defaultValue: false, description: 'Run terraform plan for secure stack (requires AWS credentials)')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Security Scan - Secure Terraform') {
      steps {
        dir('terraform') {
          sh 'trivy config --skip-dirs insecure --severity HIGH,CRITICAL --exit-code 1 --format table .'
        }
      }
    }

    stage('Terraform Plan') {
      when {
        expression { return params.RUN_TERRAFORM_PLAN }
      }
      steps {
        dir('terraform') {
          withCredentials([usernamePassword(credentialsId: 'aws-creds', usernameVariable: 'AWS_ACCESS_KEY_ID', passwordVariable: 'AWS_SECRET_ACCESS_KEY')]) {
            withEnv(["AWS_DEFAULT_REGION=us-east-1", "AWS_REGION=us-east-1"]) {
              sh 'terraform init -backend=false'
              sh 'terraform validate'
              sh 'terraform plan -input=false -lock=false -var="allowed_ssh_cidr=${ADMIN_CIDR}"'
            }
          }
        }
      }
    }
  }

  post {
    success {
      echo 'Pipeline completed successfully. Secure Terraform scan has zero critical/high findings.'
    }
    failure {
      echo 'Pipeline failed because secure Terraform scan and/or plan did not pass.'
    }
  }
}
