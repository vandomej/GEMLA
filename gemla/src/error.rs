use thiserror::Error;
use log::error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    FileLinked(file_linked::Error),
    #[error(transparent)]
    IO(std::io::Error),
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

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Error {
        Error::IO(error)
    }
}

pub fn log_error<T>(result: Result<T, Error>) -> Result<T, Error> {
    result.map_err(|e| {
        error!("{}", e);
        e
    })
}

