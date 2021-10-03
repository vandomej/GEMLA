use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    FileLinked(file_linked::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<file_linked::Error> for Error {
    fn from(error: file_linked::Error) -> Error {
        match error {
            file_linked::Error::Other(e) => Error::Other(e),
            _ => Error::FileLinked(error),
        }
    }
}
