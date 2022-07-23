#include <Eigen/Dense>

Eigen::Matrix<float, 1000, 12> compute_dw_hat(Eigen::Matrix<float, 1000, 1000> L,
 Eigen::Matrix<float, 1000, 1> theta,
 Eigen::Matrix<float, 2, 6> Js_hat,
 Eigen::Matrix<float, 2, 1> kesi_x,
 Eigen::Matrix<float, 6, 1> kesi_rall,
 Eigen::Matrix<float, 6, 7> J_pinv,
 Eigen::Matrix<float, 7, 1> kesi_q,
 Eigen::Matrix<float, 6, 12> kesi_x_prime){
    Eigen::Matrix<float, 1000, 6> dW_hat1;
    dW_hat1 = - L * theta * (Js_hat.transpose() * kesi_x + kesi_rall + J_pinv.transpose() * kesi_q).transpose();  // (1000, 6)
    Eigen::Matrix<float, 1000, 12> dW_hat;
    dW_hat = dW_hat1 * kesi_x_prime;  // (1000, 12)
    return dW_hat;
 }