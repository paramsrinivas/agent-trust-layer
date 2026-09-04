from models import PaymentAuthorizationDecision, PolicyResult, BehaviorResult

decision = PaymentAuthorizationDecision(
    request_id="REQ-DEMO",
    agent_id="AGENT-DEMO",
    session_id="SESSION-DEMO",
    decision="ALLOW",
    policy=PolicyResult(passed=True),
    behavior=BehaviorResult(anomaly_score=0.0, is_anomalous=False, features={}),
    reasons=["demo"],
)

assert decision.razorpay_mode == "test"
assert decision.razorpay_order_id is None
print("OK:", decision.razorpay_mode, decision.razorpay_order_id)
