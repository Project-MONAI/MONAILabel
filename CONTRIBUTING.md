<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  * [Signing your work](#signing-your-work)

## Introduction


Welcome to Project MONAILabel! We're excited you're here and want to contribute. This documentation is intended for individuals and institutions interested in contributing to MONAILabel. MONAILabel is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

### Communicate with us

We are happy to talk with you about your needs for MONAILabel and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point. If you are looking for an issue to resolve that will help Project MONAILabel, see the [*good first issue*](https://github.com/Project-MONAI/MONAILabel/labels/good%20first%20issue) and [*Contribution wanted*](https://github.com/Project-MONAI/MONAILabel/labels/Contribution%20wanted) labels.

## The contribution process

>In progress.  Please wait for more instructions to follow

  - Before submitting Pull Request make sure basic CI checks are passed.

    Install `pre-commit` if you have not already.
    ```
    python -m pip install pre-commit
    ```
    Run `pre-commit` check from MonaiLabel working directory.
    ```
    cd MonaiLabel
    python -m pre_commit run --all-files
    ```
    Run additional checks.
    ```
    ./runtests.sh --codeformat
    ./runtests.sh --unittests
    ```
    Run integration checks.
    ```
    export PATH=$PATH:`pwd`/monailabel/scripts
    ./runtests.sh --net
    ```


### Signing your work
MONAILabel enforces the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) on all pull requests.
All commit messages should contain the `Signed-off-by` line with an email address. The [GitHub DCO app](https://github.com/apps/dco) is deployed on MONAILabel. The pull request's status will be `failed` if commits do not contain a valid `Signed-off-by` line.

Git has a `-s` (or `--signoff`) command-line option to append this automatically to your commit message:
```bash
git commit -s -m 'a new commit'
```
The commit message will be:
```
    a new commit

    Signed-off-by: Your Name <yourname@example.org>
```

Full text of the DCO:
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
