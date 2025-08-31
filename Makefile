.PHONY: help rau-remotes rau-init rau-add rau-pull rau-push rau-backup-split

# Manage bdusell/rau as a git subtree under src/rau
# Change variables via CLI, e.g.:
#   make rau-pull RAU_BRANCH=dev

RAU_PREFIX       ?= src/rau
RAU_UPSTREAM     ?= rau_upstream
RAU_FORK         ?= rau_fork
RAU_UPSTREAM_URL ?= https://github.com/bdusell/rau.git
RAU_FORK_URL     ?= https://github.com/agiats/rau.git
RAU_BRANCH       ?= main
GIT              ?= git

help:
	@echo "Rau subtree targets:"
	@echo "  make rau-init          # add remotes for upstream and your fork"
	@echo "  make rau-add           # first-time add (requires empty $(RAU_PREFIX))"
	@echo "  make rau-pull          # pull upstream changes into subtree"
	@echo "  make rau-push          # push local subtree changes to your fork"
	@echo "  make rau-backup-split  # create branch with history of $(RAU_PREFIX)"

rau-remotes:
	@$(GIT) remote -v | grep -E '$(RAU_UPSTREAM)|$(RAU_FORK)' || true

rau-init:
	@$(GIT) remote get-url $(RAU_UPSTREAM) >/dev/null 2>&1 || $(GIT) remote add $(RAU_UPSTREAM) $(RAU_UPSTREAM_URL)
	@$(GIT) remote get-url $(RAU_FORK) >/dev/null 2>&1 || $(GIT) remote add $(RAU_FORK) $(RAU_FORK_URL)
	@echo "Remotes configured:"
	@$(MAKE) rau-remotes

# First-time add of rau as a subtree under $(RAU_PREFIX)
# Fails if $(RAU_PREFIX) already exists and is non-empty
rau-add:
	@if [ -d "$(RAU_PREFIX)" ] && [ "$$({ ls -A $(RAU_PREFIX) 2>/dev/null || true; } | wc -l)" -gt 0 ]; then \
		echo "ERROR: $(RAU_PREFIX) already exists and is not empty. Aborting."; \
		echo "Hint: commit your work, backup (make rau-backup-split), then remove $(RAU_PREFIX) before 'make rau-add'"; \
		exit 1; \
	fi
	@$(GIT) fetch $(RAU_UPSTREAM)
	@$(GIT) subtree add --prefix $(RAU_PREFIX) $(RAU_UPSTREAM) $(RAU_BRANCH) --squash

# Pull latest from upstream into subtree
rau-pull:
	@$(GIT) fetch $(RAU_UPSTREAM)
	@$(GIT) subtree pull --prefix $(RAU_PREFIX) $(RAU_UPSTREAM) $(RAU_BRANCH) --squash

# Push local subtree changes to your fork
rau-push:
	@$(GIT) fetch $(RAU_FORK) >/dev/null 2>&1 || true
	@$(GIT) subtree push --prefix $(RAU_PREFIX) $(RAU_FORK) $(RAU_BRANCH)

# Create a branch containing only the history of $(RAU_PREFIX)
rau-backup-split:
	@if [ -z "$$($(GIT) rev-list -1 HEAD -- $(RAU_PREFIX) 2>/dev/null)" ]; then \
		echo "No history found for $(RAU_PREFIX); nothing to backup. Skipping."; \
		exit 0; \
	fi
	@branch=rau_backup_split_$$(date +%Y%m%d%H%M%S); \
	$(GIT) subtree split --prefix $(RAU_PREFIX) -b $$branch >/dev/null; \
	echo "Created branch '$$branch' containing history for $(RAU_PREFIX)."; \
	echo "Push it with: git push $(RAU_FORK) $$branch:backup/$$(date +%Y%m%d%H%M%S)"


