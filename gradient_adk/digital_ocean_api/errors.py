from typing import Optional


class DOAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        payload: Optional[dict] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class DOAPIAuthError(DOAPIError): ...


class DOAPIRateLimitError(DOAPIError): ...


class DOAPIClientError(DOAPIError): ...


class DOAPIServerError(DOAPIError): ...


class DOAPINetworkError(DOAPIError): ...


class DOAPIValidationError(DOAPIError): ...
