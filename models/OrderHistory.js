const mongoose = require('./connection')

const orderHistorySchema = new mongoose.Schema({
    userid : String,
    stripe_customer : String,
    orders : [
        {
            cartItems : [],
            stripe_session_object : {},
            stripe_session_success : {},
            createdAt : Date
        }
    ]
})

const orderHistory = mongoose.model('OrderHistory', orderHistorySchema)
module.exports = orderHistory;
