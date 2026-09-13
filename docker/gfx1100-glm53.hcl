# Use with docker/docker-bake-rocm.hcl. Always build this checkout's sources.
variable "GFX1100_IMAGE" {
  default = "local/vllm:glm53-gfx1100"
}

target "glm53-gfx1100" {
  inherits = ["_common-rocm", "_labels"]
  target = "vllm-openai"
  args = {
    ARG_PYTORCH_ROCM_ARCH = "gfx1100"
    REMOTE_VLLM = "0"
    NIC_BACKEND = "none"
  }
  tags = [GFX1100_IMAGE]
  output = ["type=docker"]
}
