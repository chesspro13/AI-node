VERSION="0.1.0"

docker build . -t  "chesspro13/ai-node"
docker build . -t  "chesspro13/ai-node:v${VERSION}"

docker push chesspro13/ai-node
docker push "chesspro13/ai-node:v${VERSION}"