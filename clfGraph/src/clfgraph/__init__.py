import sentry_sdk


sentry_sdk.init(
    dsn="https://d4953e4c6eab9d20a73ea6e0dff1731e@o4507599957786624.ingest.de.sentry.io/4507861198700624",
    enable_tracing=True,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
    release="clfgraph@0.2.0",
    max_breadcrumbs=50,
    debug=True,
)