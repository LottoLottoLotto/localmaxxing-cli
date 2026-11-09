# Build from the repository root:
#   docker build -f cmd/lmx-relay/Dockerfile -t lmx-relay .
# Base images are pinned by digest; bump the tag and digest together
# (docker buildx imagetools inspect <image:tag>). The module stays `go 1.22`
# for compatibility; the image builds with a supported Go release.
FROM golang:1.27.3-alpine@sha256:d530cf60723abf1ffae0cd16bb546b379ec0550d2f3815e5b16ed69dbc00fba4 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY internal ./internal
COPY cmd/lmx-relay ./cmd/lmx-relay
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lmx-relay ./cmd/lmx-relay

FROM gcr.io/distroless/static-debian13:nonroot@sha256:d62067879de890a1def9da0e7766b7d20810a5e5555cef8506b752437d7a3e88
COPY --from=build /out/lmx-relay /lmx-relay
USER nonroot:nonroot
# 8080: public (agents, /healthz; route through the reverse proxy).
# 8081: internal (/internal/*; private network only, never published).
EXPOSE 8080 8081
ENV LMX_RELAY_ADDR=:8080 LMX_RELAY_INTERNAL_ADDR=:8081
ENTRYPOINT ["/lmx-relay"]
