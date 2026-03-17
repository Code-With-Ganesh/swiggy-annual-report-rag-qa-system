pipeline {
  agent any

  options {
    ansiColor('xterm')
    timestamps()
  }

  parameters {
    booleanParam(name: 'RUN_INSECURE_BASELINE', defaultValue: false, description: 'Run intentional vulnerable baseline scan for assignment evidence (expected UNSTABLE).')
    string(name: 'ADMIN_CIDR', defaultValue: '203.0.113.10/32', description: 'Trusted CIDR for SSH in secure Terraform stack')
    booleanParam(name: 'RUN_TERRAFORM_PLAN', defaultValue: false, description: 'Run terraform plan for secure stack (requires AWS credentials)')
    string(name: 'AWS_REGION', defaultValue: 'us-east-1', description: 'AWS region for Terraform and deploy stages')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Infrastructure Security Scan - Insecure Baseline') {
      when {
        expression { return params.RUN_INSECURE_BASELINE }
      }
      steps {
        dir('terraform/insecure') {
          script {
            // Expected assignment behavior: fail/warn here, continue pipeline for remediation.
            catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
              sh 'trivy config --severity HIGH,CRITICAL --exit-code 1 --format json --output trivy-insecure-report.json .'
            }
            sh 'trivy config --severity HIGH,CRITICAL . || true'
          }
        }
      }
    }

    stage('AI Risk Analysis + Remediation Suggestions') {
      when {
        expression { return params.RUN_INSECURE_BASELINE }
      }
      steps {
        sh 'python3 scripts/ai_remediate.py --report terraform/insecure/trivy-insecure-report.json --secure-file terraform/main.tf --insecure-file terraform/insecure/main.tf'
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
            withEnv(["AWS_DEFAULT_REGION=${params.AWS_REGION}", "AWS_REGION=${params.AWS_REGION}"]) {
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
    always {
      archiveArtifacts artifacts: 'terraform/insecure/trivy-insecure-report.json', allowEmptyArchive: true
    }
    success {
      echo 'Pipeline completed successfully. Secure Terraform scan has zero critical/high findings.'
    }
    unstable {
      echo 'Pipeline is unstable because insecure baseline scan found vulnerabilities (expected in assignment mode).'
    }
    failure {
      echo 'Pipeline failed. Check scan/plan/deploy stage logs for the exact issue.'
    }
  }
}
