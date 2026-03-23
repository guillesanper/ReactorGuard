# backend.tf
# Configura el backend remoto de Terraform en Google Cloud Storage (GCS).
# Almacenar el estado de forma remota permite colaboración en equipo y
# evita conflictos cuando varias personas aplican cambios simultáneamente.
# GCS implementa locking automático mediante generaciones de objetos,
# por lo que no es necesario configurar una tabla DynamoDB como en AWS.

terraform {
  backend "gcs" {
    # Bucket donde se almacenará el fichero terraform.tfstate
    bucket = "reactorguard-terraform-state"

    # Prefijo dentro del bucket que separa el estado de cada entorno.
    # El fichero real quedará en:
    #   gs://reactorguard-terraform-state/terraform/state/dev/default.tfstate
    prefix = "terraform/state/dev"
  }
}
