# Cargo Registry Login Action

A GitHub Action that authenticates to one or more Cargo registry using provided tokens. 

## Why Environment Variables?

This action uses environment variables for token configuration as opposed to running `cargo login` or saving the credentials to a file because:
- They are seamlessly integrated with both `cargo` and `cross` commands
- They provide a secure way to handle authentication tokens
- They persist throughout the workflow execution

## Usage

Add the action to your workflow and provide the registry tokens as input. The tokens should be in key=value format, with one token per line.

#### Login to crates.io

To authenticate with crates.io, provide your token in the following format:

```yaml
- name: Login to Cargo registry
  uses: p6m-dev/github-actions/cargo-registry-login@main
  with:
    cargo-tokens: |
      crates-io=your-crates-io-token-here
```

#### Multiple Registry Authentication

When working with multiple registries, you can provide tokens for each registry on separate lines:

```yaml
- name: Login to multiple Cargo registries
  uses: p6m-dev/github-actions/cargo-registry-login@main
  with:
    cargo-tokens: |
      crates-io=your-crates-io-token-here
      my-private-registry=private-registry-token
      another-registry=another-token
```

#### Using GitHub Secrets

For security, it's recommended to store your tokens as GitHub Secrets:

```yaml
- name: Login to Cargo registries
  uses: p6m-dev/github-actions/cargo-registry-login@main
  with:
    cargo-tokens: |
      crates-io=${{ secrets.CRATES_IO_TOKEN }}
      private-registry=${{ secrets.PRIVATE_REGISTRY_TOKEN }}
```


### Inputs

| Input | Description | Required |
|-------|-------------|----------|
| `cargo-tokens` | Registry tokens in key=value format (one per line) | No |

### Token Format

The `cargo-tokens` input accepts tokens in the following format:
- One registry token per line
- Each line should be in `registry=token` format
- Use `crates-io` for the official crates.io registry
- Use custom names for other registries

### Environment Variables Set

For crates.io:
- `CARGO_REGISTRY_CREDENTIAL_PROVIDER=cargo:token`
- `CARGO_REGISTRY_TOKEN=Bearer <token>`

For custom registries:
- `CARGO_REGISTRIES_<REGISTRY>_CREDENTIAL_PROVIDER=cargo:token`
- `CARGO_REGISTRIES_<REGISTRY>_TOKEN=Bearer <token>`

Note: Custom registry names are converted to uppercase and hyphens are replaced with underscores.
