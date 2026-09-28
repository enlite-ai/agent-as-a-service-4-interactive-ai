# syntax=docker/dockerfile:1

# Which policy serves PowerGrid: "xd" or "t2.1". Both ship in this image
# regardless of this value - it only sets the container's default, which an
# `-e A3S_POWERGRID_AGENT=...` at `docker run`/compose time still overrides. See
# the README section "Choosing which agent serves PowerGrid".
ARG A3S_POWERGRID_AGENT=t2.1

FROM python:3.11.6-slim

ARG A3S_POWERGRID_AGENT
ENV A3S_POWERGRID_AGENT=${A3S_POWERGRID_AGENT}

# Install the (large: torch/grid2op/lightsim2grid) Python deps before copying the
# rest of the source, so this slow layer is cached and only rebuilt when
# pyproject.toml actually changes. The pip cache mount persists downloaded
# wheels across builds, so a retry after a network blip does not re-download.
# Both the base deps and the t2.1 extra install in one pip call (not separate
# COPY+RUN layers) so pip's resolver sees every constraint together — nothing
# installed second can silently upgrade a package (e.g. numpy) pinned by the
# other set. Nothing from the t2.1 extra is imported until its first request.
WORKDIR /my_app
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --default-timeout=2000 \
      --extra-index-url https://download.pytorch.org/whl/cpu ".[t2-1]"

COPY . .

CMD ["bash", "./docker/entrypoint.sh"]
