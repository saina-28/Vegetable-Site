const mongoose = require('./connection')

const orderProcessingSchema = new mongoose.Schema({
    userid : String,
    stripe_customer : String,
    cartItems : [{
        itemid : String,
        item_name : String,
        quantity : Number,
        item_price : Number,
    }],
    stripe_session_object : {}
})

const OrderProcessing = mongoose.model('OrderProcessing', orderProcessingSchema)

module.exports = OrderProcessing;
