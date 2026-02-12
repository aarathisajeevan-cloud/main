import re

class fake_words:
    def __init__(self):
        pass
    def has_payment_request(self, text):
        text = text.lower()

        patterns = [
            # --- Direct payment words ---
            r'payment',
            r'fee',
            r'fees',
            r'charge',
            r'charges',
            r'amount',
            r'cost',

            # --- Registration / training scams ---
            r'registration',
            r'register',
            r'enrollment',
            r'enrolment',
            r'training fee',
            r'onboarding fee',
            r'joining fee',
            r'processing fee',

            # --- Deposits ---
            r'deposit',
            r'security deposit',
            r'refundable',

            # --- Currency symbols & formats ---
            r'rs\.?\s?\d+',
            r'inr\s?\d+',
            r'\$\s?\d+',

            # --- UPI / bank hints ---
            r'upi',
            r'google pay',
            r'phonepe',
            r'paytm',
            r'bank transfer',
            r'account number',
            r'ifsc',

            # --- Scam phrasing ---
            r'pay before',
            r'pay to confirm',
            r'payment required',
            r'payment mandatory',
            r'fees required',
            r'fees mandatory',
            r'pay and join'
        ]

        matches = []
        for p in patterns:
             if re.search(p, text):
                 matches.append(p.replace(r'\s?', ' ').replace(r'\d+', '#').replace('\\', ''))

        return bool(matches), matches

    def suspicious_easy_job(self, text):
        text = text.lower()

        phrases = [
            "no experience needed",
            "no experience required",
            "freshers can apply",
            "anyone can apply",
            "easy job",
            "simple work",
            "no interview",
            "direct joining",
            "immediate joining",
            "guaranteed job",
            "guaranteed placement",
            "work from home and earn",
            "high salary without experience"
        ]

        matches = [p for p in phrases if p in text]
        return bool(matches), matches

