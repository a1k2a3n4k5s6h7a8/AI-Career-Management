// script.js
// Client-side validation, password matching checks, and dynamic form features.

document.addEventListener("DOMContentLoaded", function () {
    'use strict';

    // -------------------------------------------------------------
    // 1. BOOTSTRAP 5 FORM VALIDATION
    // -------------------------------------------------------------
    // Fetch all forms we want to apply custom Bootstrap validation styles to
    const forms = document.querySelectorAll('.needs-validation');

    // Loop over them and prevent submission on invalid inputs
    Array.from(forms).forEach(function (form) {
        form.addEventListener('submit', function (event) {
            // Special custom checks
            let customValid = true;

            // If it is the registration form, check if passwords match
            if (form.id === 'registerForm') {
                const password = document.getElementById('password');
                const confirmPassword = document.getElementById('confirm_password');
                const feedback = document.getElementById('confirmPasswordFeedback');

                if (password.value !== confirmPassword.value) {
                    confirmPassword.setCustomValidity("Passwords do not match.");
                    feedback.textContent = "Passwords do not match.";
                    customValid = false;
                } else {
                    confirmPassword.setCustomValidity("");
                }
            }

            // Standard HTML5 validation check
            if (!form.checkValidity() || !customValid) {
                event.preventDefault();
                event.stopPropagation();
            }

            form.classList.add('was-validated');
        }, false);
    });

    // -------------------------------------------------------------
    // 2. LIVE PASSWORD MATCH CHECKING (REAL-TIME FEEDBACK)
    // -------------------------------------------------------------
    const passwordInput = document.getElementById('password');
    const confirmInput = document.getElementById('confirm_password');
    const registerForm = document.getElementById('registerForm');

    if (passwordInput && confirmInput && registerForm) {
        const checkPasswordMatch = function () {
            const feedback = document.getElementById('confirmPasswordFeedback');
            if (confirmInput.value === "") {
                confirmInput.setCustomValidity("Please confirm your password.");
                feedback.textContent = "Please confirm your password.";
            } else if (passwordInput.value !== confirmInput.value) {
                confirmInput.setCustomValidity("Passwords do not match.");
                feedback.textContent = "Passwords do not match.";
            } else {
                confirmInput.setCustomValidity("");
                feedback.textContent = "";
            }
        };

        passwordInput.addEventListener('input', checkPasswordMatch);
        confirmInput.addEventListener('input', checkPasswordMatch);
    }

    // -------------------------------------------------------------
    // 3. ASSESSMENT FORM NUMERICAL RANGE BOUNDARY VALIDATION
    // -------------------------------------------------------------
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        // Enforce logical range values for ML features
        const cgpa = document.getElementById('cgpa');
        const attendance = document.getElementById('attendance_percentage');
        
        if (cgpa) {
            cgpa.addEventListener('input', function() {
                const val = parseFloat(cgpa.value);
                if (val < 0 || val > 10) {
                    cgpa.setCustomValidity("CGPA must be between 0.00 and 10.00.");
                } else {
                    cgpa.setCustomValidity("");
                }
            });
        }

        if (attendance) {
            attendance.addEventListener('input', function() {
                const val = parseFloat(attendance.value);
                if (val < 0 || val > 100) {
                    attendance.setCustomValidity("Attendance percentage must be between 0 and 100.");
                } else {
                    attendance.setCustomValidity("");
                }
            });
        }
    }
});
