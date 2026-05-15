param location string
param tags object
param name string
param environmentId string
param registryServer string
param identityId string
param identityClientId string
param imageName string
param targetPort int = 8501
param cpu string = '1.0'
param memory string = '2.0Gi'
param minReplicas int = 1
param maxReplicas int = 3

param aoaiEndpoint string
param aoaiDeploymentName string

param useWhisper bool = false
param whisperEndpoint string = ''
param whisperDeploymentName string = ''
@secure()
param whisperApiKey string = ''

var baseEnv = [
  {
    name: 'AZURE_OPENAI_ENDPOINT'
    value: aoaiEndpoint
  }
  {
    name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
    value: aoaiDeploymentName
  }
  {
    name: 'USE_WHISPER'
    value: string(useWhisper)
  }
  {
    // Hint to DefaultAzureCredential to pick the right managed identity.
    name: 'AZURE_CLIENT_ID'
    value: identityClientId
  }
]

var whisperBaseEnv = useWhisper ? [
  {
    name: 'WHISPER_ENDPOINT'
    value: whisperEndpoint
  }
  {
    name: 'WHISPER_DEPLOYMENT_NAME'
    value: whisperDeploymentName
  }
] : []

// Only inject the secret-backed env var when an API key was actually provided.
// Otherwise the app authenticates to Whisper via Managed Identity (DefaultAzureCredential).
var whisperKeyEnv = useWhisper && !empty(whisperApiKey) ? [
  {
    name: 'WHISPER_API_KEY'
    secretRef: 'whisper-api-key'
  }
] : []

var whisperEnv = concat(whisperBaseEnv, whisperKeyEnv)

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityId}': {}
    }
  }
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: registryServer
          identity: identityId
        }
      ]
      secrets: useWhisper && !empty(whisperApiKey) ? [
        {
          name: 'whisper-api-key'
          value: whisperApiKey
        }
      ] : []
    }
    template: {
      containers: [
        {
          name: 'app'
          image: imageName
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: concat(baseEnv, whisperEnv)
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
    }
  }
}

output id string = app.id
output name string = app.name
output fqdn string = app.properties.configuration.ingress.fqdn
output identityId string = identityId
