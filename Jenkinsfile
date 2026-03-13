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

    stage('Infrastructure Security Scan - Insecure Baseline') {
      steps {
        dir('terraform/insecure') {
          script {
            // Keep pipeline moving to AI analysis while still surfacing a failing scan.
            catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
              sh 'trivy config --severity HIGH,CRITICAL --exit-code 1 --format json --output trivy-insecure-report.json .'
            }
            sh 'trivy config --severity HIGH,CRITICAL . || true'
          }
        }
      }
    }

    stage('AI Risk Analysis + Remediation Suggestions') {
      steps {
        sh 'python3 scripts/ai_remediate.py --report terraform/insecure/trivy-insecure-report.json --secure-file terraform/main.tf --insecure-file terraform/insecure/main.tf'
      }
    }

    stage('Security Scan - Secure Terraform') {
      steps {
        dir('terraform') {
          sh 'trivy config --severity HIGH,CRITICAL --exit-code 1 --format table .'
        }
      }
    }

    stage('Terraform Plan') {
      when {
        expression { return params.RUN_TERRAFORM_PLAN }
      }
      steps {
        dir('terraform') {
          sh 'terraform init -backend=false'
          sh 'terraform validate'
          sh 'terraform plan -input=false -lock=false -var="allowed_ssh_cidr=${ADMIN_CIDR}"'
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
      echo 'Pipeline is unstable because insecure baseline scan found vulnerabilities (expected in pre-remediation stage).'
    }
  }
}
