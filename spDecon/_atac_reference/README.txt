Private compatibility backend for the fixed-reference ATAC paper protocol.

The original module and class names are retained for numerical traceability.
The public wrapper enables only count-only, fixed-reference, three-initializer fitting.
No HMRF affinity or spatial prior is active in this protocol. Historical HMRF helpers
are not exposed through the public spDecon API. The public core and spatial models
live outside this compatibility package.
