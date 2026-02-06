from enum import Enum


class PayoffType(Enum):
    CALL = "Call"
    PUT = "Put"
    ASIAN_CALL = "Asian_Call"
    ASIAN_PUT = "Asian_Put"
    DIGITAL_CALL = "Digital_Call"
    DIGITAL_PUT = "Digital_Put"

    PUT_MIN = "PUT_MIN"
    PUT_MAX = "PUT_MAX"
    PUT_AVERAGE = "PUT_AVERAGE"

    CALL_MIN = "CALL_MIN"
    CALL_MAX = "CALL_MAX"
    CALL_AVERAGE = "CALL_AVERAGE"
